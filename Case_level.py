import os
import json
from tools.VQA import VQA_Module
from MOE_Decider import MOE_Decider
from Summary_Module import Summary_Module
from Evaluator import Evaluator

OPENAI_API_KEY = ""

## Load data
data_root = "Fundus"
full_record = os.path.join(data_root,'full_record')
brief_record = os.path.join(data_root,'brief_record')
pred_record = os.path.join(data_root,'pred_record')

img_dir = "/mnt/data0/ziyue/dataset/Glaucoma/REFUGE2/Training400"
name_list = [
            f"Glaucoma_{file}" for file in os.listdir(os.path.join(img_dir, 'Glaucoma'))
        ] + [
            f"Non-Glaucoma_{file}" for file in os.listdir(os.path.join(img_dir, 'Non-Glaucoma'))
        ]

## Prompt format
prompt1 = "Please describe the observations made in the fundus image."
prompt2 = "Does the patient have Peripapillary Atrophy according to the fundus image? Elaborate on your answer and support with visual evidence from the image."
prompt3 = "Please describe the observations made in the fundus image and provide a diagnosis on optic drance hemorrhages."


vqa = VQA_Module("Glaucoma")
summary = Summary_Module(OPENAI_API_KEY)

# weights = {"cdr": 0.3, "rt": 0.3, "ppa": 0.2}
weights = {"ppa": 0.2}
decider = MOE_Decider(weights, 0.4)
evaluator = Evaluator("moe_prediction")

for idx in range(3):
    example = name_list[idx]
    subdir, file = example.split('_')
    image_path = os.path.join(img_dir, subdir,file)
    full_json = os.path.join(full_record, example.split(".")[0] + ".json")
    brief_json = os.path.join(brief_record, example.split(".")[0] + ".json")
    pred_json = os.path.join(pred_record, example.split(".")[0] + ".json")

    answer = vqa.get_answer(image_path, prompt2)
    vqa.save_answer(full_json, "ppa", answer)

    summary_text = summary.summarize(full_json, brief_json, "ppa")

    metrics = decider.decide(brief_json, pred_json)

evaluator.evaluate(pred_record)