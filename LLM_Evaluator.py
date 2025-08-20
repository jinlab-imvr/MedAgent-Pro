import os
import json
import openai

class LLM_Evaluator:
    def __init__(self, api_key):
        """
        Initialize the LLM_Decider object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI 的 API Key
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def evaluate(self,  output_file, prediction, answer, field):
        """
        Evaluate the output of the LLM model based on the ground truth answer.

        Args:
            output_file (str): output file path
            prediction (str): prediction of the LLM model
            answer (str): ground truth answer

        Returns:
            dict: result of the LLM model
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please evaluate the following prediction."},
            {"role": "user", "content": f"The model's prediction is {prediction} \
                \nPlease evaluate the prediction based on the following ground truth answer: {answer} \
                \nIs the prediction correct? Answer with only one word (Yes or No)"}
        ]

        completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="chatgpt-4o-latest",
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