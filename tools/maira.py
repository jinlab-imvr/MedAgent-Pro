import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class MAIRA:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    # def predict(self, image_path, phrase, output_file, field):
    #     image = Image.open(image_path)
    #     processed_inputs = self.processor.format_and_preprocess_phrase_grounding_input(
    #         frontal_image=image,
    #         phrase=phrase,
    #         return_tensors="pt",
    #     )
    #     processed_inputs = processed_inputs.to(self.device)
    #     with torch.no_grad():
    #         output_decoding = self.model.generate(
    #             **processed_inputs,
    #             max_new_tokens=150,
    #             use_cache=True,
    #         )
    #     prompt_length = processed_inputs["input_ids"].shape[-1]
    #     decoded_text = self.processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
        
    #     try:
    #         prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    #     except (ValueError, AssertionError) as e:
    #         prediction = [(None, None)]

    #     if prediction[0][1] is None:
    #         bbox = None
    #     else:
    #         bbox = prediction[0][1][0]
    #         width, height = image.size
    #         bbox = self.processor.adjust_box_for_original_image_size(bbox, width, height)
   
    #     if os.path.exists(output_file):
    #         with open(output_file, "r", encoding="utf-8") as json_file:
    #             existing_data = json.load(json_file)
    #     else:
    #         existing_data = {}
    #     key = field if field else "janus_prediction"
    #     existing_data[key] = bbox

    #     with open(output_file, "w", encoding="utf-8") as json_file:
    #         json.dump(existing_data, json_file, indent=4)


    #     return bbox
    def predict(self, image_path, phrase, output_file, field):
        image = Image.open(image_path)

        processed_inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,  # Our example has no prior
            indication="Lung Leision",
            technique="PA and lateral views of the chest.",
            comparison="None",
            prior_report=None,  # Our example has no prior
            return_tensors="pt",
            get_grounding=True,  # For this example we generate a non-grounded report
        )

        processed_inputs = processed_inputs.to(self.device)
        with torch.no_grad():
            output_decoding = self.model.generate(
                **processed_inputs,
                max_new_tokens=300,  # Set to 450 for grounded reporting
                use_cache=True,
            )



        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = self.processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
        decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
        # prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)   
        try:
            prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
        except (ValueError, AssertionError) as e:
            prediction = None

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        key = field if field else "maira_prediction"
        existing_data[key] = prediction

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)


        return prediction