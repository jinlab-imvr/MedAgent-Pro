import os
from RAG import RAG_Module

OPENAI_API_KEY = ""

## Load data
data_root = "../Fundus"
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

rag = RAG_Module(openai_api_key=OPENAI_API_KEY)
query_text = "how to diagnose glaucoma"
answer = rag.query(query_text)
print("Answer:")
print(answer)



