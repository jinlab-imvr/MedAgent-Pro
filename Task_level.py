import os
import json

from RAG import RAG_Module
from Planner import Planner

OPENAI_API_KEY = ''

rag = RAG_Module(openai_api_key=OPENAI_API_KEY)
planner = Planner(api_key=OPENAI_API_KEY)
 
## Load data
data_root = "./Glaucoma"
task_file = os.path.join(data_root, 'task.json')
tool_file = os.path.join(data_root, 'toolset.json')
brief_file = os.path.join(data_root,'brief_record/REFUGE')
pred_dir = os.path.join(data_root,'brief_record/REFUGE')

img_dir = "/mnt/data0/ziyue/dataset/Glaucoma/REFUGE2/Training400"
name_list = [
            f"Glaucoma_{file}" for file in os.listdir(os.path.join(img_dir, 'Glaucoma'))
        ] + [
            f"Non-Glaucoma_{file}" for file in os.listdir(os.path.join(img_dir, 'Non-Glaucoma'))
        ]

example = name_list[0]
subdir, file = example.split('_')
image_path = os.path.join(img_dir, subdir,file)


rag_prompt = "how to diagnose glaucoma"
rag_result = rag.query(rag_prompt)
print("RAG result:")
print(rag_result)

# 1) load task description
with open(task_file, "r", encoding="utf-8") as f:
    task = json.load(f)

input_desc   = str(task.get("input", "")).strip()
disease_goal = str(task.get("disease", "")).strip()

# 2) load toolset.json
with open(tool_file, "r", encoding="utf-8") as f:
    toolset = json.load(f)

# 3) build planner prompt (English, strict fields)
planner_prompt = (
    "Plan a step-by-step, executable workflow using ONLY the available tools.\n"
    f"Input: {input_desc}\n"
    f"Goal: {disease_goal}\n"
    "Output format (STRICT): an array of objects with fields "
    "[id, tool, action_type, action, input_type, output_type, output_path].\n"
    "- id starts from 1 and increases by 1\n"
    "- tool is an ARRAY of integers (tool ids from the toolset)\n"
    "- action_type is a STRING: 'qualitative' or 'quantitative'\n"
    "- input_type is an ARRAY of integers; use 0 for raw/original inputs, or a prior step's id if the input is that step's output\n"
    "- The field output_type MUST be EXACTLY one of: 'intermediate result' or 'final indicator'\n"
    "- For any non-image output, set output_path EXACTLY to 'diagnosis.json'; only images may use distinct file paths\n"
    "Return ONLY the JSON array."
)

planner.plan(data_root, planner_prompt, rag_result, filename="plan.json", toolset=toolset)

