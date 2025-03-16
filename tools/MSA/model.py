import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image

from . import cfg
from .utils import *

class SAM_Adapter:
    def __init__(self, sam_ckpt, weights):
        self.args = cfg.parse_args()
        self.GPUdevice = torch.device('cuda', self.args.gpu_device)
        self.args.sam_ckpt = sam_ckpt
        self.args.weights = weights
        self.net = get_network(self.args, self.args.net, use_gpu=self.args.gpu, gpu_device=self.GPUdevice, distribution = args.distributed)

        print(os.path.abspath(self.args.weights))
        print(f'=> resuming from {self.args.weights}')
        assert os.path.exists(self.args.weights)
        checkpoint_file = os.path.join(self.args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(self.args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)


        state_dict = checkpoint['state_dict']
        if self.args.distributed != 'none':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k[7:] # remove `module.`
                name = 'module.' + k
                new_state_dict[name] = v
            # load params
        else:
            new_state_dict = state_dict

        self.net.load_state_dict(new_state_dict,strict=False)
        self.net.eval()

    def click_prompt(self, mask, point_labels = 1, category=None):
        # check if all masks are black
        max_label = max(set(mask.flatten()))
        if max_label == 0:
            point_labels = max_label
        # max agreement position
        indices = np.argwhere(mask == category) 
        return point_labels, indices[np.random.randint(len(indices))]

    def predict_mask(self,img_path,save_path,category=1):
        img = Image.open(img_path).convert('RGB')
        ori_shape = np.array(img).shape
        transform= transforms.Compose([
            transforms.Resize((self.args.image_size,self.args.image_size)),
            transforms.ToTensor(),
        ])

        mask_dir = "/mnt/data0/ziyue/dataset/Glaucoma/REFUGE2/Annotation-Training400/Disc_Cup_Masks/Glaucoma"
        mask = cv2.imread(os.path.join(mask_dir,os.path.basename(img_path).replace(".jpg",".bmp")),0)

        point_labels = 1
        point_labels, pt = self.click_prompt(mask, point_labels=1, category=category)

        # pt = [[512,256],[512,512],[512,768]]
        img = transform(img).unsqueeze(0)
        img = img.to(dtype = torch.float32, device = self.GPUdevice)

        point_labels = np.array([point_labels])
        pt = np.array([pt])
        print(pt)
        # box = torch.tensor([10,10,1013,1013]).to(dtype = torch.float32, device = self.GPUdevice).unsqueeze(0).unsqueeze(0)
        # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
        point_coords = pt
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.GPUdevice)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.GPUdevice)


        if(len(point_labels.shape)==1): # only one point prompt
            coords_torch, labels_torch= coords_torch[None, :, :], labels_torch[None, :]
        pt = (coords_torch, labels_torch)
        with torch.no_grad():
            imge= self.net.image_encoder(img)
            se, de = self.net.prompt_encoder(
                                points=pt,
                                boxes=None,
                                masks=None,
                            )
            pred, _ = self.net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=self.net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                        
        # Resize to the ordered output size
        pred = F.interpolate(pred,size=(ori_shape[0],ori_shape[1]))
        pred = torch.sigmoid(pred).detach()
        pred = (pred>0.5).float()
        # pred = 1-pred
        pred = pred[0][0].detach().cpu().numpy()
        cv2.imwrite(save_path,pred*255)
        return pred
