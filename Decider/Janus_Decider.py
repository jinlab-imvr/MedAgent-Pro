import os
import json
import torch
from PIL import Image
import io
import base64
from transformers import AutoModelForCausalLM
import tempfile

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def concat_pil_images_horizontally(pil_images):
    """
    将多个 PIL.Image 横向拼接成一张图像

    Args:
        pil_images (List[PIL.Image.Image]): 图片列表

    Returns:
        PIL.Image.Image: 拼接后的图片
    """
    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)
    new_image = Image.new('RGB', (total_width, max_height))
    
    current_x = 0
    for img in pil_images:
        new_image.paste(img, (current_x, 0))
        current_x += img.width
    return new_image

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

    def decide(self, output_file, prompt, image_paths=None, field=None):
        """
        Decide the output of the Janus model based on the prompt.

        Args:
            output_file (str): Output file path.
            prompt (str): Prompt for the model.
            image_paths (str or list, optional): Single image path or a list of image paths.
            field (str, optional): Field name for saving the result. Default key is "janus_prediction".

        Returns:
            str: The prediction result.
        """
        # 如果 image_paths 存在且不是列表，则转换为列表
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\n" + prompt,
                "images": image_paths if image_paths else []
            },
            {"role": "Assistant", "content": ""}
        ]

        # 加载 PIL 图像
        pil_images = load_pil_images(conversation)
        # 如果返回的图像数量大于1，则进行横向拼接
        if len(pil_images) > 1:
            concatenated_image = concat_pil_images_horizontally(pil_images)
            pil_images = [concatenated_image]
        
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,  # 指定生成新令牌的数量
            do_sample=True,
            temperature=0.7,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 读取或初始化输出数据
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}
        key = field if field else "janus_prediction"
        existing_data[key] = answer

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

        return answer
