import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class MAIRA:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, phrase, output_file, field):
        processed_inputs = self.processor.format_and_preprocess_phrase_grounding_input(
            frontal_image=image,
            phrase=phrase,
            return_tensors="pt",
        )
        processed_inputs = processed_inputs.to(self.device)
        with torch.no_grad():
            output_decoding = self.model.generate(
                **processed_inputs,
                max_new_tokens=150,
                use_cache=True,
            )
        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = self.processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
        prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

        bbox = prediction[0][1][0]

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        key = field if field else "janus_prediction"
        existing_data[key] = bbox

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)


        return bbox