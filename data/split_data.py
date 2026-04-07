import json

with open("data/data.jsonl") as f:
    lines = [json.loads(l) for l in f]

split = int(len(lines) * 0.9)
train, val = lines[:split], lines[split:]

# Save train split
with open("data/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")

# Save test inputs (no fields/output)
with open("data/test.jsonl", "w") as f:
    for r in val:
        f.write(json.dumps({"input": r["input"], "format": r["format"]}) + "\n")

# Save ground truth (no input/output)
with open("data/test_ground_truth.jsonl", "w") as f:
    for r in val:
        f.write(json.dumps({"fields": r["fields"], "format": r["format"]}) + "\n")

print(f"Train: {len(train)}, Test: {len(val)}")