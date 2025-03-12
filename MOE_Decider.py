import os
import json
import numpy as np
from sklearn.metrics import f1_score

class MOE_Decider:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def compute_score(self, entry):
        pos_score = sum(self.weights[key] if "yes" in entry[key].lower() else 0.15 for key in self.weights)
        return pos_score

    def evaluate(self, data_root):
        files = sorted(os.listdir(data_root))
        all_labels = []
        all_predict = []
        
        for file in files:
            # Determine the true label based on the file name
            label = 1 if file.startswith("Glaucoma") else 0
            all_labels.append(label)
            
            # Load JSON data from file
            file_path = os.path.join(data_root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Compute score and generate prediction based on the threshold
            score = self.compute_score(data)
            prediction = 1 if score >= self.threshold else 0
            all_predict.append(prediction)
        
        # Calculate accuracy metrics
        pos_example = all_labels.count(1)
        neg_example = all_labels.count(0)
        
        pos_acc = np.sum((np.array(all_labels) == 1) & (np.array(all_predict) == 1))
        neg_acc = np.sum((np.array(all_labels) == 0) & (np.array(all_predict) == 0))
        
        mAcc = (pos_acc / pos_example + neg_acc / neg_example) / 2
        f1 = f1_score(all_labels, all_predict)
        
        # Print results
        print(f"Glaucoma Accuracy: {pos_acc}")
        print(f"Non-Glaucoma Accuracy: {neg_acc}")
        print(f"Mean Accuracy: {mAcc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Return the results as a dictionary
        return {
            "Glaucoma Accuracy": pos_acc,
            "Non-Glaucoma Accuracy": neg_acc,
            "Mean Accuracy": mAcc,
            "F1 Score": f1
        }

# Example usage:
weights = {"cdr": 0.3, "rt": 0.3, "ppa": 0.2}
decider = MOE_Decider(weights, 0.5)
metrics = decider.evaluate("../Fundus/brief_record/REFUGE")
