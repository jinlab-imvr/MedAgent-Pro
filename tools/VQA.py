import os
import json
from tools.VisionUnite.model import VisionUniteModel

class VQA_Module:
    def __init__(self, disease_type):
        if disease_type == "Glaucoma":
            self.ckpt_path = "/mnt/data0/ziyue/MedAgent/VisionUnite/checkpoint"
            self.model = VisionUniteModel(self.ckpt_path)
        else:
            print("Please input the correct disease type.")
            return None  
    
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