import os
import torch
from PIL import Image
from transformers import set_seed, logging

# 关闭 transformers 日志输出
logging.set_verbosity_error()

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

class LLaVaMed:
    def __init__(
        self,
        model_path,
        model_base=None,
        conv_mode="vicuna_v1",
        temperature=0.2,
        top_p=None,
        num_beams=1,
    ):
        """
        Initialize the LLaVaMed model.

        Args:
            model_path (str): checkpoint, like "microsoft/llava-med-v1.5-mistral-7b"
            model_base (str, optional): model base, default to None
            conv_mode (str): conversation template mode, default to "vicuna_v1"
            temperature (float): temperature for generation
            top_p (float): top_p sampling parameter
            num_beams (int): beam search number
        """

        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 复制对话模板
        self.conv = conv_templates[self.conv_mode].copy()

    def get_answer(self, image_path, prompt):
        if self.model.config.mm_use_im_start_end:
            prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conversation_history = [{"role": self.conv.roles[0], "content": prompt_text}]

        self.conv.messages = []
        for msg in conversation_history:
            self.conv.append_message(msg["role"], msg["content"])
        self.conv.append_message(self.conv.roles[1], None)

        full_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        img = Image.open(image_path).convert("RGB")
        image_tensor = process_images([img], self.image_processor, self.model.config)[0].unsqueeze(0).half().to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=1024,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs