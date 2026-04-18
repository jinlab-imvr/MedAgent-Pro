import os
import json
import torch
import base64

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



class Qwen_Decider:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_path,min_pixels=min_pixels, max_pixels=max_pixels)
        self.model.eval()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def decide(self, output_file, prompt, image_paths=None, field=None):
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_messages = []
        if image_paths:
            for path in image_paths:
                base64_image = self.encode_image(path)
                image_messages.append({
                    "type": "image",
                    "image": f"data:image;base64,{base64_image}"
                })

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please help me make a decision based on the following information."},
            {
                "role": "user",
                "content": image_messages+[
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data[field] = output_text
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return output_text
