"""End-to-end evaluation entry point.

Loads a submission (LoRA weights + optional generate.py), runs inference
on the test set, scores the predictions, and prints results.

Usage:
    python -m evaluate.run_eval --submission_dir path/to/submission \
        [--test_path data/test.jsonl] \
        [--ground_truth_path data/test_ground_truth.jsonl] \
        [--output_path results.json]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.scoring import score_all
from inference.baseline_generate import generate as baseline_generate, load_model


def load_custom_generate(generate_path: str):
    """Dynamically import a participant's generate function."""
    spec = importlib.util.spec_from_file_location("custom_generate", generate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate"):
        raise ValueError(
            f"Submission generate.py at {generate_path} must define a "
            "'generate(model, tokenizer, prompt, format_name) -> str' function"
        )
    return module.generate


def detect_lora_path(submission_dir: str) -> str | None:
    """Check if the submission directory contains LoRA adapter files."""
    sub_path = Path(submission_dir)
    # Look for adapter_config.json (PEFT standard)
    if (sub_path / "adapter_config.json").exists():
        return str(sub_path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission")
    parser.add_argument(
        "--submission_dir",
        type=str,
        required=True,
        help="Path to submission directory",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "test.jsonl"),
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "test_ground_truth.jsonl"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "output" / "results.json"),
    )
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    if not sub_dir.exists():
        print(f"Error: submission directory not found: {sub_dir}")
        sys.exit(1)

    # Detect LoRA weights
    lora_path = detect_lora_path(args.submission_dir)
    if lora_path:
        print(f"Found LoRA adapter at: {lora_path}")
    else:
        print("No LoRA adapter found — using base model")

    # Detect custom generate function
    generate_py = sub_dir / "generate.py"
    if generate_py.exists():
        print(f"Found custom generate.py at: {generate_py}")
        generate_fn = load_custom_generate(str(generate_py))
    else:
        print("No custom generate.py — using baseline generation")
        generate_fn = baseline_generate

    # Load model
    model, tokenizer = load_model(lora_path)

    # Load test data
    print(f"Loading test data from: {args.test_path}")  # This will be later substituted for our test sample
    test_samples = []
    with open(args.test_path, encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))

    # Run inference
    print(f"Running inference on {len(test_samples)} samples...")
    predictions = []
    for i, sample in enumerate(test_samples):
        pred = generate_fn(model, tokenizer, sample["input"], sample["format"])
        predictions.append({
            "input": sample["input"],
            "format": sample["format"],
            "prediction": pred,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_samples)} done")

    # Load ground truth
    ground_truths = []
    with open(args.ground_truth_path, encoding="utf-8") as f:
        for line in f:
            ground_truths.append(json.loads(line))

    # Score
    results = score_all(predictions, ground_truths)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall score: {results['overall']:.4f}")
    print(f"Valid outputs: {results['valid_count']}/{results['total_samples']} "
          f"({results['valid_ratio']:.1%})")
    print("\nPer-format scores:")
    for fmt, score in sorted(results["per_format"].items()):
        print(f"  {fmt:>5s}: {score:.4f}")
    print("=" * 50)

    # Save detailed results
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
