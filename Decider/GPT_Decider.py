import os
import json
import openai
import base64

class GPT_Decider:
    def __init__(self, api_key):
        """
        Initialize the LLM_Decider object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI 的 API Key
        """
        self.api_key = api_key
        openai.api_key = self.api_key
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def decide(self, output_file, prompt, image_paths=None, field=None):
        """
        Decide the output of the LLM model based on the prompt.

        Args:
            output_file (str): output file path
            prompt (str): prompt for the LLM model

        Returns:
            dict: result of the LLM model
        """
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_messages = []
        if image_paths:
            for path in image_paths:
                base64_image = self.encode_image(path)
                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                })

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please help me make a decision based on the following information."},
            {"role": "user", "content": image_messages + [{
                "type": "text",
                "text": prompt,
            }]}
        ]

        completion = openai.ChatCompletion.create(
            model="chatgpt-4o-latest",
            # model = "gpt-4o",
            # model = "o1",
            messages=messages
        )
        result = completion.choices[0].message.content

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        existing_data[field] = result
        
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return existing_data