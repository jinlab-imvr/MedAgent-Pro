import os
import json
from tools.VisionUnite.model import VisionUniteModel
from tools.GPT_VQA import GPT_VQA

class VQA_Module:
    def __init__(self, vqa_module, model_path):
        if vqa_module.lower() == "glaucoma":  # Specialized tools for glaucoma
            print("Using VisionUnite model as the VQA module.")
            self.ckpt_path = "/mnt/data0/ziyue/MedAgent/VisionUnite/checkpoint"
            self.model = VisionUniteModel(self.ckpt_path)
        elif vqa_module.lower() == "llavamed":  # Specialized tools for llavamed
            from tools.LLaVaMed.model import LLaVaMed
            print("Using LLaVaMed model as the VQA module.")
            self.ckpt_path = "microsoft/llava-med-v1.5-mistral-7b" # load from huggingface
            self.model = LLaVaMed(self.ckpt_path)
        else:
            self.ckpt_path = model_path
            self.model = GPT_VQA(self.ckpt_path)
            print("Using GPT-4o as the VQA module.")
    
    def get_answer(self, image_path, prompt):
        answer = self.model.get_answer(image_path, prompt)
        return answer

    def save_answer(self, json_file, indicator, answer):
        if not os.path.exists(json_file):
            with open(json_file, "w", encoding="utf-8") as file:
                json.dump({}, file, indent=4)
                
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        data[indicator] = answer

        with open(json_file, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)