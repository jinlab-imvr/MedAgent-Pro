import os
import json
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

class Janus_Decider:
    def __init__(self, model_path):
        """
        Initialize the Janus_Decider object with the model path.

        Args:
            model_path (str): Path to the Janus model.
        """
        self.model_path = model_path
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def decide(self, output_file, prompt, image_path=None):
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + prompt,
                "images": [image_path],
            },
            {"role": "Assistant", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data["janus_prediction"] = answer
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return answer