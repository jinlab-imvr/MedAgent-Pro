import os
from skimage import io, transform
import numpy as np
from .segment_anything import sam_model_registry
import torch
import torch.nn.functional as F

def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().detach().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

class MedSAM:
    def  __init__(self, checkpoint, device=None):
        self.checkpoint = checkpoint
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sam_model_registry["vit_b"](checkpoint=self.checkpoint)
        self.model.to(device=self.device)
        self.model.eval()

    def predict_mask(self, input_path, box, output_path):

        img_np = io.imread(input_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        
        H, W, _ = img_3c.shape
        img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        box_np = np.array([[int(x) for x in box[1:-1].split(',')]]) 
        # transfer box_np t0 1024x1024 scale
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            image_embedding = self.model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

        medsam_seg = medsam_inference(self.model, image_embedding, box_1024, H, W)
        io.imsave(
            output_path,
            medsam_seg,
            check_contrast=False,
        )
