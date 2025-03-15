import os
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
from . import llama

class VisionUniteModel:
    def __init__(self, data_dir, device=None):
        """
        Initialize VisionUnite model
        
        Args:
            data_dir (str): Directory containing LLaMA directory and weight file
            device (str, optional): Specify device, such as "cuda" or "cpu". If not passed, it will be automatically detected.
        """
        self.data_dir = data_dir
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        llama_dir = os.path.join(self.data_dir, "llama")
        weight_path = os.path.join(self.data_dir, "checkpoint-VisionUniteV1.pth")
        
        # 加载模型
        self.model = llama.load(weight_path, llama_dir).to(self.device)
        self.model.eval()
        self.model.to(self.device)
    
    def _load_and_transform_vision_data(self, image_paths):
        """
        Load and preprocess vision data        
        Args:
            image_paths (list of str): List of image file paths
            
        Returns:
            torch.Tensor: (N, C, H, W)
        """
        if image_paths is None:
            return None

        image_outputs = []
        for image_path in image_paths:
            data_transform = transforms.Compose([
                transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            with open(image_path, "rb") as fopen:
                image = Image.open(fopen).convert("RGB")
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.3)
                image = np.array(image)
                min_R = np.min(image[:, :, 0])
                min_G = np.min(image[:, :, 1])
                min_B = np.min(image[:, :, 2])
                image[:, :, 0] = image[:, :, 0] - min_R + 1
                image[:, :, 1] = image[:, :, 1] - min_G + 1
                image[:, :, 2] = image[:, :, 2] - min_B + 1
                image = Image.fromarray(image.astype('uint8')).convert('HSV')
            
            image = data_transform(image).to(self.device)
            image_outputs.append(image)
        return torch.stack(image_outputs, dim=0)
    
    def get_answer(self, image_path, prompt):
        """
        Get answer from VisionUnite model
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Prompt for the model
            
        Returns:
            str: Answer from the model
        """
        sample = [image_path]
        prompts = [prompt]
        input_data = self._load_and_transform_vision_data(sample)
        results, cls_pred = self.model.generate(input_data, prompts, input_type="vision")
        return results[0]


