"""Example custom generation function.

Participants can modify this file to implement their own generation strategy.
The function signature must be exactly:

    generate(model, tokenizer, prompt: str, format_name: str) -> str

This example adds a format-specific system prompt, which can improve output
quality compared to the generic baseline prompt.
"""

import torch


def generate(model, tokenizer, prompt: str, format_name: str) -> str:
    """Generate structured output with format-specific prompting.

    Args:
        model: The language model (base or with LoRA merged).
        tokenizer: The tokenizer.
        prompt: Natural language input describing the data.
        format_name: Target format ("json", "yaml", "xml", "csv", "toml").

    Returns:
        The generated structured output as a string.
    """
    format_hints = {
        "json": "Output valid JSON object. Use double quotes for keys and string values.",
        "yaml": "Output valid YAML. Use key: value format, one per line.",
        "xml": "Output valid XML with <record> as root element and field names as child tags.",
        "csv": "Output valid CSV with a header row and one data row.",
        "toml": "Output valid TOML. Use key = value format, quote string values.",
    }

    system_content = (
        f"You convert natural language descriptions into {format_name.upper()} format. "
        f"{format_hints.get(format_name, '')} "
        "Output only the formatted data, nothing else."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()
