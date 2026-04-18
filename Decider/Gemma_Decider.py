from transformers import pipeline
from PIL import Image
import requests
import torch
import os
import json

class Gemma_Decider:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pipeline(
            "image-text-to-text",
            model=self.model_path,
            torch_dtype=torch.bfloat16,
            device=self.device,
        )

    def encode_image(self, image_path):
        image = Image.open(image_path)
        return image

    def decide(self, output_file, prompt, image_paths=None, field=None):
        if image_paths:
            image = self.encode_image(image_paths)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Please help me make a decision based on the following information."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        output = self.model(text=messages, max_new_tokens=200)
        output_text = output[0]["generated_text"][-1]["content"]

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data[field] = output_text
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return output_text