# coding_agent.py
import os
import re
import openai
from typing import Optional

class Coding_Agent:
    def __init__(self, api_key: str):
        """
        Args:
            api_key (str): OpenAI API Key
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    # ---------- prompt assembly ----------
    def _build_messages(
        self,
        requirement: str,
        enforce_function_name: Optional[str] = None,
        extra_context: Optional[str] = None,
    ):
        """
        Build messages for ChatCompletion using your dialog format.
        The model must return ONLY a single complete Python function (no fences).
        """
        system_msg = (
            "You are a Python coding assistant. "
            "Return ONLY a single COMPLETE Python function definition. "
            "No markdown fences, no explanations, no tests. "
            "Keep it self-contained (import inside the function if needed), "
            "avoid print statements, add a brief docstring."
        )

        user_text = (
            "Requirement:\n"
            f"{requirement}\n\n"
        )
        if extra_context:
            user_text += "Additional context:\n" + str(extra_context).strip() + "\n\n"
        if enforce_function_name:
            user_text += (
                "Use EXACTLY this function name:\n"
                f"{enforce_function_name}\n\n"
            )
        user_text += (
            "Output strictly the function code only. "
            "Do not include any text outside the function."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]
        return messages

    # ---------- parsing ----------
    def _strip_fences(self, text: str) -> str:
        t = (text or "").strip()
        # remove common ```python ... ``` wrappers if any
        if t.startswith("```"):
            parts = t.split("```")
            # pick the longest code-looking chunk
            parts = sorted((p.strip() for p in parts), key=len, reverse=True)
            for p in parts:
                if p.startswith("def "):
                    return p
            # fallback
            return parts[0] if parts else ""
        return t

    def _extract_function_name(self, code: str) -> str:
        """
        Extract function name from 'def name(...):'
        """
        m = re.search(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code, flags=re.MULTILINE)
        if not m:
            raise ValueError("Could not find a Python function signature in the model output.")
        return m.group(1)

    def _append_to_file(self, file_path: str, code: str):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        needs_gap = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        with open(file_path, "a", encoding="utf-8") as f:
            if needs_gap:
                f.write("\n\n")
            f.write(code.rstrip() + "\n")

    # ---------- public API ----------
    def generate_function(
        self,
        output_file: str,
        requirement: str,
        model: str = "chatgpt-4o-latest",
        enforce_function_name: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Generate a function per requirement, append to output_file (no overwrite),
        and return the function name.

        Args:
            output_file (str): target .py file to append the function into
            requirement (str): natural-language requirement/spec
            model (str): OpenAI model name
            enforce_function_name (str|None): if provided, the model must use this exact name
            extra_context (str|None): optional hints (I/O spec, examples, constraints)

        Returns:
            str: generated function name
        """
        messages = self._build_messages(
            requirement=requirement,
            enforce_function_name=enforce_function_name,
            extra_context=extra_context,
        )

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        raw = completion.choices[0].message.content or ""
        code = self._strip_fences(raw)

        fn_name = self._extract_function_name(code)
        if enforce_function_name and fn_name != enforce_function_name:
            raise ValueError(f"function_name must be '{enforce_function_name}', got '{fn_name}'.")

        self._append_to_file(output_file, code)
        return fn_name
