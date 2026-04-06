"""Baseline generation / inference code for the structured output task.

Loads Qwen 3.5-0.8B (optionally with LoRA), runs inference on test data,
and writes predictions to a JSONL file.

Usage:
    python -m inference.baseline_generate [--lora_path PATH] [--test_path PATH] [--output_path PATH]
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_NEW_TOKENS = 512

# System prompt used in chat template
SYSTEM_PROMPT = (
    "You convert natural language descriptions into structured data formats. "
    "Output only the formatted data, nothing else."
)

def load_model(lora_path: str = None) -> tuple:
    """Load base model, optionally merge LoRA weights.

    Returns (model, tokenizer).
    """
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path is not None:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("  LoRA merged successfully")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, format_name: str) -> str:
    """Generate structured output for a single prompt.

    This is the function participants can override in their submission.

    Args:
        model: The language model (with or without LoRA merged).
        tokenizer: The tokenizer.
        prompt: The natural language input (e.g. "JSON; name is John Doe").
        format_name: The target format (e.g. "json", "yaml", "xml", "csv", "toml").

    Returns:
        The generated structured output as a string.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy — participants can experiment with sampling
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()


def run_inference(
    model,
    tokenizer,
    test_path: str,
    output_path: str,
) -> None:
    """Run inference on all test samples and write predictions."""
    print(f"Loading test data from: {test_path}")
    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Running inference on {len(samples)} samples...")
    predictions = []
    for i, sample in enumerate(samples):
        pred = generate(model, tokenizer, sample["input"], sample["format"])
        predictions.append({
            "input": sample["input"],
            "format": sample["format"],
            "prediction": pred,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)} done")

    print(f"Writing predictions to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Baseline inference")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "test.jsonl"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "output" / "predictions.jsonl"),
    )
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.lora_path)
    run_inference(model, tokenizer, args.test_path, args.output_path)


if __name__ == "__main__":
    main()
