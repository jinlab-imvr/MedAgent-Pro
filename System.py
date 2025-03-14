import os
from MOE_Decider import MOE_Decider
from RAG import RAG_Module

OPENAI_API_KEY = "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"

data_root = "../Fundus"
brief_file = os.path.join(data_root,'brief_record/REFUGE')
pred_dir = os.path.join(data_root,'brief_record/REFUGE')

rag = RAG_Module(openai_api_key=OPENAI_API_KEY)
query_text = "how to diagnose glaucoma"
answer = rag.query(query_text)
print("Answer:")
print(answer)

weights = {"cdr": 0.3, "rt": 0.3, "ppa": 0.2}
decider = MOE_Decider(weights, 0.4)
metrics = decider.evaluate(brief_file)

