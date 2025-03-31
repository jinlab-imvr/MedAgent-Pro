import os
import json
import numpy as np
from sklearn.metrics import f1_score

class MOE_Decider:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def compute_score(self, entry):
        pos_score = sum(
            self.weights[key] if "yes" in entry[key].lower() 
            else (0 if "no" in entry[key].lower() else self.weights[key] / 2)
            for key in self.weights
        )
        return pos_score

    def decide(self, brief_file_path, pred_file_path):        
        with open(brief_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        score = self.compute_score(data)
        prediction = 1 if score >= self.threshold else 0
                
        output_dir = os.path.dirname(pred_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(pred_file_path):
            with open(pred_file_path, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data["moe_prediction"] = prediction

        with open(pred_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        
        return existing_data

