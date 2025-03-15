import json
import os
import torchvision.models
from pathlib import Path
import numpy as np
import timm
import open_clip

import copy
import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.checkpoint as cp
from torchvision.transforms import Resize

from timm.models.layers import trunc_normal_
from functools import partial
import timm.models.vision_transformer

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
import cv2

from .Config import get_CTranS_config
from .llama import Transformer, ModelArgs
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download
from .LViT import LViT

KEYWORD_DICT = {
    1: "Other abnormalities. ",
    2: "Hemorrhages exudation abnormalities. ",
    3: "Optic cup disc abnormalities. ",
    4: "Color boundary abnormalities. ",
    5: "Macular abnormalities. ",
    6: "Arteriovenous abnormalities. ",
    0: "Overall normality. "
    }

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class EVA02(nn.Module):
    def __init__(self, model):
        super(EVA02, self).__init__()
        self.model = timm.create_model(model, pretrained=False, num_classes=0) # feture size=768
    def forward(self, x):
        x = self.model(x)
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class LLaMA_adapter(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, llama_ckpt_dir, llama_tokenizer, knn=False):
        super().__init__()
        
        model = EVA02('eva02_base_patch14_448')
        self.logit_scale_init_value = 0.07
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))
        # model.head = nn.Linear(num_ftrs, 4096)  # 7B
        # model.head = nn.Linear(num_ftrs, 5120)  # 13B
        self.eva02 = model
        self.adapter = nn.Linear(768, 4084)  # with clip
        
        self.adapter_text = nn.Linear(192, 768)
        # self.image_bind = imagebind_model.imagebind_huge(pretrained=True)

        self.abnormal1 = nn.Linear(4084, 2)  # 7B
        self.abnormal2 = nn.Linear(4084, 2)  # 7B
        self.abnormal3 = nn.Linear(4084, 2)  # 7B
        self.abnormal4 = nn.Linear(4084, 2)  # 7B
        self.abnormal5 = nn.Linear(4084, 2)  # 7B
        self.abnormal6 = nn.Linear(4084, 2)  # 7B
        
        self.transforms = T.Resize(size = (224,224))
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

        # 2. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 3. llama
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=512, max_batch_size=32, **params
        )
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
        """
        loaded = [
            torch.load(ckpt_shared, map_location='cpu') 
            for ckpt_shared in ckpts
        ]
        for param_tensor in loaded[0].keys():
            ckpt = {param_tensor: loaded[0][param_tensor]}
            try:
                if loaded[0][param_tensor].size()[0] != self.llama.state_dict()[param_tensor].size()[0]:
                    ckpt = {param_tensor: torch.cat((loaded[0][param_tensor],loaded[1][param_tensor]),dim=0)}
                elif loaded[0][param_tensor].size()[1] != self.llama.state_dict()[param_tensor].size()[1]:
                    ckpt = {param_tensor: torch.cat((loaded[0][param_tensor],loaded[1][param_tensor]),dim=1)}
            except:
                if loaded[0][param_tensor].size()[0] != self.llama.state_dict()[param_tensor].size()[0]:
                    ckpt = {param_tensor: torch.cat((loaded[0][param_tensor],loaded[1][param_tensor]),dim=0)}
        """
        self.llama.load_state_dict(ckpt, strict=False)

        # 4. prefix
        self.query_layer = 32  # 7B
        # self.query_layer = 40   # 13B
        self.query_len = 1
        self.prefix_query = nn.Embedding(self.query_layer * self.query_len, model_args.dim)
        
        # 5. knn
        self.knn = knn
        if knn:
            import faiss
            self.index = faiss.read_index("/path_to_knn_index/knn.index")
        
        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_ab = torch.nn.CrossEntropyLoss()
        # self.criterion_ab = torch.nn.CrossEntropyLoss(reduction="none")
        # self.criterion_cls = torch.nn.functional.cross_entropy(reduction='none')
        self.set_default_trainability()

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()
    
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            # trainable[name] = para
            if name.startswith("llama."):
                if 'norm' in name or 'bias' in name or 'lora' in name:
                    trainable[name] = para
            # elif name.startswith("eva02."):
            #     if 'adapter' in name:
            #         trainable[name] = para
            # elif name.startswith("LViT."):
            #     trainable[name] = para
            else:
                trainable[name] = para
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True

    def forward_visual(self, imgs, input_type):
        # visual_feats = self.image_bind({input_type : imgs})[input_type]
        # self.eva02.load_state_dict(torch.load('/cpfs01/user/lizihan/modality-classification/eva02-cls-7-92.pth'), strict=False)
        visual_feats = self.eva02(imgs)
        device = visual_feats.device
        return visual_feats
    
    def clip_encode_image(self, x):
        # from CLIP
        x = self.clip(x)
        feature_map = x[0]
        # x = torch.as_tensor(x_temp).to(x.device)
        feature_map = torch.cat([fm.unsqueeze(0) for fm in feature_map], dim=0)
        # print(feature_map.shape)
        return feature_map

    @torch.inference_mode()
    def forward_inference(self, visual_feats, tokens, start_pos: int):
        
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)
        prefix_query = self.prefix_query.weight.reshape(
            self.query_layer, 1, 4096).unsqueeze(1)
        prefix_index = 0
        visual_proj = visual_feats.unsqueeze(1)
        for layer in self.llama.layers[-1 * self.query_layer:]:
            h = layer(h, start_pos, freqs_cis, mask, visual_proj + prefix_query[prefix_index].repeat(_bsz, 1, 1))
            prefix_index = prefix_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    def forward(self, tokens, labels, imgs, cls_label, Keyword, target):
        for i in range(len(Keyword)):
            Keyword[i] = torch.tensor(self.tokenizer.encode(Keyword[i], bos=True, eos=False), dtype=torch.int64)
            padding = 192 - Keyword[i].shape[0]
            if padding > 0:
                Keyword[i] = torch.cat((Keyword[i], torch.zeros(padding, dtype=torch.int64)+1))
            else:
                Keyword[i] = Keyword[i][:192]
        Keyword = torch.stack(Keyword, dim=0).to(imgs.device).to(torch.float32)
        visual_feats = self.forward_visual(imgs, "vision")

        local_clip_image_feats = self.clip_encode_image(self.transforms(imgs))
        visual_feats = visual_feats + local_clip_image_feats

        text_feats = self.adapter_text(Keyword)
        text_feats = F.normalize(text_feats)
        logits_per_image = self.compute_logits(visual_feats, text_feats)
        logits_per_text = logits_per_image.t()

        # Compute cross-entropy loss
        clip_loss = self.softce_clip_loss(logits_per_text, target)
        
        visual_feats = self.adapter(visual_feats)
        abnormal_feats1 = self.abnormal1(visual_feats)
        abnormal_feats2 = self.abnormal2(visual_feats)
        abnormal_feats3 = self.abnormal3(visual_feats)
        abnormal_feats4 = self.abnormal4(visual_feats)
        abnormal_feats5 = self.abnormal5(visual_feats)
        abnormal_feats6 = self.abnormal6(visual_feats)
        
        abnormal_feats = torch.cat((abnormal_feats1, abnormal_feats2, abnormal_feats3, abnormal_feats4, abnormal_feats5, abnormal_feats6), dim=1)
        
        visual_feats = torch.cat((visual_feats, abnormal_feats), dim=1)

        visual_feats = visual_feats.half()

        cls_loss = self.criterion_ab(abnormal_feats1, cls_label[0])
        cls_loss += self.criterion_ab(abnormal_feats2, cls_label[1])
        cls_loss += self.criterion_ab(abnormal_feats3, cls_label[2])
        cls_loss += self.criterion_ab(abnormal_feats4, cls_label[3])
        cls_loss += self.criterion_ab(abnormal_feats5, cls_label[4])
        cls_loss += self.criterion_ab(abnormal_feats6, cls_label[5])
            
        # add keyword token
        Keyword_temp = torch.rand(1, 48).to(imgs.device)
        labels_temp = torch.rand(1, 48).to(imgs.device)
        for i in range(len(abnormal_feats1)):
            Keyword_text = ""
            cls_pred_1 = abnormal_feats1[i].argmax(dim=-1)
            cls_pred_2 = abnormal_feats2[i].argmax(dim=-1)
            cls_pred_3 = abnormal_feats3[i].argmax(dim=-1)
            cls_pred_4 = abnormal_feats4[i].argmax(dim=-1)
            cls_pred_5 = abnormal_feats5[i].argmax(dim=-1)
            cls_pred_6 = abnormal_feats6[i].argmax(dim=-1)
            
            if cls_pred_1 > 0:
                Keyword_text += KEYWORD_DICT[1]
            if cls_pred_2 > 0:
                Keyword_text += KEYWORD_DICT[2]
            if cls_pred_3 > 0:
                Keyword_text += KEYWORD_DICT[3]
            if cls_pred_4 > 0:
                Keyword_text += KEYWORD_DICT[4]
            if cls_pred_5 > 0:
                Keyword_text += KEYWORD_DICT[5]
            if cls_pred_6 > 0:
                Keyword_text += KEYWORD_DICT[6]
            if not ((cls_pred_1 > 0) or (cls_pred_2 > 0) or (cls_pred_3 > 0) or (cls_pred_4 > 0) or (cls_pred_5 > 0) or (cls_pred_6 > 0)):
                Keyword_text = KEYWORD_DICT[0]
            Keyword_text = torch.tensor(self.tokenizer.encode(Keyword_text, bos=True, eos=False), dtype=torch.int64)
            padding = 48 - Keyword_text.shape[0]
            if padding > 0:
                Keyword_text = torch.cat((Keyword_text, torch.zeros(padding, dtype=torch.int64)-1))
            else:
                Keyword_text = Keyword_text[:48]
            Keyword_text = Keyword_text.ge(0)
            Keyword_text[~Keyword_text] = 0
            Keyword_text = torch.unsqueeze(Keyword_text,dim=0).to(imgs.device)
            Keyword_label = torch.unsqueeze(torch.zeros(48, dtype=torch.int64),dim=0).to(imgs.device)
            Keyword_temp = torch.cat((Keyword_temp, Keyword_text), dim=0)
            labels_temp = torch.cat((labels_temp, Keyword_label), dim=0)
            
        tokens = torch.cat((Keyword_temp[0:-1].long(), tokens), dim=1)
        labels = torch.cat((labels_temp[0:-1].long(), labels), dim=1)
        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)
        prefix_query = self.prefix_query.weight.reshape(
            self.query_layer, 1, 4096).unsqueeze(1)
        # prefix_query = self.prefix_query.weight.reshape(
        #     self.query_layer, 1, 5120).unsqueeze(1)
        prefix_index = 0
        visual_proj = visual_feats.unsqueeze(1)

        for layer in self.llama.layers[-1 * self.query_layer:]:
            h = layer(h, 0, freqs_cis, mask, visual_proj + prefix_query[prefix_index])
            prefix_index = prefix_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())
        
        return c_loss, cls_loss, clip_loss

    @torch.inference_mode()
    def generate(
            self,
            imgs,
            prompts,
            input_type,
            max_gen_len: int = 256,
            temperature: float = 0.1,
            top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs, input_type)
            local_clip_image_feats = self.clip_encode_image(self.transforms(imgs))  
            visual_query = visual_query + local_clip_image_feats
            visual_query = self.adapter(visual_query) # with clip
            cls_pred = []

            # abnormal_feats = self.abnormal(visual_query)
            abnormal_feats1 = self.abnormal1(visual_query)
            abnormal_feats2 = self.abnormal2(visual_query)
            abnormal_feats3 = self.abnormal3(visual_query)
            abnormal_feats4 = self.abnormal4(visual_query)
            abnormal_feats5 = self.abnormal5(visual_query)
            abnormal_feats6 = self.abnormal6(visual_query)
        
            abnormal_feats = torch.cat((abnormal_feats1, abnormal_feats2, abnormal_feats3, abnormal_feats4, abnormal_feats5, abnormal_feats6), dim=1)
            
            for i in range(len(abnormal_feats1)):
                abnormal_feats1[i] = abnormal_feats1[i].softmax(dim=-1)
                abnormal_feats2[i] = abnormal_feats2[i].softmax(dim=-1)
                abnormal_feats3[i] = abnormal_feats3[i].softmax(dim=-1)
                abnormal_feats4[i] = abnormal_feats4[i].softmax(dim=-1)
                abnormal_feats5[i] = abnormal_feats5[i].softmax(dim=-1)
                abnormal_feats6[i] = abnormal_feats6[i].softmax(dim=-1)
                cls_pred_1 = 1 if abnormal_feats1[i][1] >= abnormal_feats1[i][0] else 0
                cls_pred_2 = 1 if abnormal_feats2[i][1] >= abnormal_feats1[i][0] else 0
                cls_pred_3 = 1 if abnormal_feats3[i][1] >= abnormal_feats1[i][0] else 0
                cls_pred_4 = 1 if abnormal_feats4[i][1] >= abnormal_feats1[i][0] else 0
                cls_pred_5 = 1 if abnormal_feats5[i][1] >= abnormal_feats1[i][0] else 0
                cls_pred_6 = 1 if abnormal_feats6[i][1] >= abnormal_feats1[i][0] else 0
                
                cls_pred.append([cls_pred_1, cls_pred_2, cls_pred_3, cls_pred_4, cls_pred_5, cls_pred_6])

            
            visual_query = torch.cat((visual_query, abnormal_feats), dim=1)
            visual_query = visual_query.half()

        Keyword_temp = []
        for i in range(len(cls_pred)):
            Keyword_text = ""
            if cls_pred[i][0] > 0:
                Keyword_text += KEYWORD_DICT[1]
            if cls_pred[i][1] > 0:
                Keyword_text += KEYWORD_DICT[2]
            if cls_pred[i][2] > 0:
                Keyword_text += KEYWORD_DICT[3]
            if cls_pred[i][3] > 0:
                Keyword_text += KEYWORD_DICT[4]
            if cls_pred[i][4] > 0:
                Keyword_text += KEYWORD_DICT[5]
            if cls_pred[i][5] > 0:
                Keyword_text += KEYWORD_DICT[6]
            if not ((cls_pred[i][0] > 0) or (cls_pred[i][1] > 0) or (cls_pred[i][2] > 0) or (cls_pred[i][3] > 0) or (cls_pred[i][4] > 0) or (cls_pred[i][5] > 0)):
                Keyword_text = KEYWORD_DICT[0]
            Keyword_text = self.tokenizer.encode(Keyword_text, bos=True, eos=False)
            Keyword_temp.append(Keyword_text)
        # padding = self.max_words - input2.shape[0]
        # if padding < 0:
        #     input2 = input2[:self.max_words]
        # print(prompts.shape[0])
        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        for i in range(len(prompts)):
            prompts[i] = Keyword_temp[i] + prompts[i]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded, cls_pred


_MODELS = {
    "7B": "https://coming_soon.pth",
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', knn=False):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}")

    llama_type = "7B"
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_ckpt_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    adapter_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model_cfg = adapter_ckpt.get('config', {})

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path, knn=knn)

    load_result = model.load_state_dict(adapter_ckpt['model'], strict=False)
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device)
