"""Validate a submission directory before submitting.

Checks that the submission has the correct structure and that LoRA weights
and/or generate.py are loadable.

Usage:
    python -m submission.validate_submission path/to/submission
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path


def validate(submission_dir: str) -> bool:
    """Validate a submission directory. Returns True if valid."""
    sub_path = Path(submission_dir)
    ok = True

    if not sub_path.exists():
        print(f"FAIL: Directory does not exist: {sub_path}")
        return False

    if not sub_path.is_dir():
        print(f"FAIL: Not a directory: {sub_path}")
        return False

    has_lora = (sub_path / "adapter_config.json").exists()
    has_generate = (sub_path / "generate.py").exists()

    if not has_lora and not has_generate:
        print("WARN: No adapter_config.json and no generate.py found.")
        print("      The base model with default generation will be used.")
        print("      This is allowed but probably not what you want.")

    # Check LoRA adapter
    if has_lora:
        print("CHECK: Found adapter_config.json")
        try:
            import json
            with open(sub_path / "adapter_config.json") as f:
                config = json.load(f)
            if "r" in config and "target_modules" in config:
                print("  OK: adapter_config.json looks valid")
            else:
                print("  WARN: adapter_config.json may be incomplete")
        except Exception as e:
            print(f"  FAIL: Could not parse adapter_config.json: {e}")
            ok = False

        # Check for adapter weights
        weight_files = list(sub_path.glob("adapter_model*"))
        if weight_files:
            print(f"  OK: Found adapter weights: {[f.name for f in weight_files]}")
        else:
            print("  WARN: adapter_config.json found but no adapter_model* files")
    else:
        print("INFO: No LoRA adapter found — base model will be used")

    # Check generate.py
    if has_generate:
        print("CHECK: Found generate.py")
        try:
            spec = importlib.util.spec_from_file_location(
                "custom_generate", str(sub_path / "generate.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "generate"):
                sig = inspect.signature(module.generate)
                params = list(sig.parameters.keys())
                if len(params) >= 4:
                    print(f"  OK: generate() function found with params: {params}")
                else:
                    print(f"  WARN: generate() has only {len(params)} params, expected 4: "
                          "(model, tokenizer, prompt, format_name)")
            else:
                print("  FAIL: generate.py does not define a 'generate' function")
                ok = False
        except ImportError as e:
            print(f"  WARN: Could not fully load generate.py (missing dependency: {e})")
            print("        This is OK if the dependency is available at evaluation time.")
            # Check if generate function exists via AST instead
            import ast
            try:
                with open(sub_path / "generate.py") as gf:
                    tree = ast.parse(gf.read())
                func_names = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]
                if "generate" in func_names:
                    print("  OK: 'generate' function found via static analysis")
                else:
                    print("  FAIL: No 'generate' function found in generate.py")
                    ok = False
            except Exception as e2:
                print(f"  FAIL: Could not parse generate.py: {e2}")
                ok = False
        except Exception as e:
            print(f"  FAIL: Could not load generate.py: {e}")
            ok = False
    else:
        print("INFO: No generate.py found — baseline generation will be used")

    # Summary
    print()
    if ok:
        print("RESULT: Submission structure is valid!")
    else:
        print("RESULT: Submission has errors — please fix them before submitting.")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate submission")
    parser.add_argument("submission_dir", type=str, help="Path to submission directory")
    args = parser.parse_args()

    valid = validate(args.submission_dir)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
