import os
import openai
import base64

# Examine if a indicator is trustworthy for medical diagnosis
class Examiner:
    def __init__(self, api_key):
        """
        Initialize the Examiner object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI 的 API Key
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.function = "Determine if the result is confidential and accurate?\
            Answer with only one word (Yes or No)"

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def examine(self, prompt, image_path=None, result=None):

        image_messages = []
        
        base64_image = self.encode_image(image_path)
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            },
        })

        if os.path.isfile(result):
            base64_image = self.encode_image(result)
            image_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            })

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please help me make decision"},
                {"role": "user", "content": image_messages + [{
                    "type": "text",
                    "text": prompt + self.function,
                }]}
            ]
        
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please determine if the result is confidential and accurate?\
                 Answer with only one word (Yes or No)"},
                {"role": "user", "content": image_messages + [{
                    "type": "text",
                    "text": prompt + self.function + "\n" + result,
                }]}
            ]
        
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        result = completion.choices[0].message.content
        
        return result
