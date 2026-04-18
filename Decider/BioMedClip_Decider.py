import os
import json
from PIL import Image
import torch

from open_clip import create_model_and_transforms, get_tokenizer, create_model_from_pretrained
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

class BioMedClip_Decider:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = 256

        self.model, self.preprocess = create_model_from_pretrained(self.model_path)
        self.tokenizer = get_tokenizer(self.model_path)

        self.model.to(self.device)
        self.model.eval()

    def decide(self, output_file, prompt, image_path=None, field=None):
        template = 'this photo related to '
        labels = [
             prompt,
            'Normal',
        ]

        img = Image.open(image_path)
        images = torch.stack([self.preprocess(img)]).to(self.device)
        texts = self.tokenizer([template + l for l in labels], context_length=self.context_length).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)

            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)

            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()

        top_k = 1
        # label = subdir
        for i in range(images.shape[0]):
            pred = labels[sorted_indices[i][0]]

            top_k = len(labels) if top_k == -1 else top_k
            for j in range(top_k):
                jth_index = sorted_indices[i][j]
        
        pred = labels[jth_index]

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        key = field if field else "janus_prediction"
        if jth_index == 0:
            existing_data[key] = "Yes"
        else:
            existing_data[key] = "No"

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return pred