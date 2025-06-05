# python src/prepare_data.py \
#   --txt_folder data/ocr_txts \
#   --json_folder data/ground_truth_json \
#   --output_path data/dataset.arrow

import os
import json
import argparse
from datasets import Dataset, DatasetDict

def load_raw_examples(txt_folder: str, json_folder: str):
    """
    Reads each .txt file in `txt_folder`, where each .txt is the OCR result
    in space-separated words. Then finds a matching ground-truth JSON (by base name)
    under `json_folder`. Returns a list of dicts containing:
      - file_name: base name (no extension)
      - ocr_text: all contents of the .txt as a single string
      - ground_truth_json: a JSON-string (pretty-printed) of the ground truth
    """
    examples = []
    for fn in os.listdir(txt_folder):
        if not fn.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fn)[0]
        txt_path = os.path.join(txt_folder, fn)

        # Read OCR text from .txt
        with open(txt_path, "r", encoding="utf-8") as f:
            ocr_text = f.read().strip()

        # Load corresponding ground-truth JSON (if present)
        json_path = os.path.join(json_folder, base + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as fj:
                gt = json.load(fj)
        else:
            gt = {}  # or raise a warning if you prefer

        # Convert ground-truth dict → pretty‐printed string
        gt_str = json.dumps(gt, ensure_ascii=False, indent=2)

        examples.append({
            "file_name": base,
            "ocr_text": ocr_text,
            "ground_truth_json": gt_str
        })
    return examples

def create_prompt_response(example: dict):
    ocr = example["ocr_text"]
    prompt = f"""
Given the OCR text below, extract the following details in well-formatted JSON exactly as shown, ensuring all "N/A" values remain unchanged.
Extract and include the following fields (placeholder names—replace with actual fields):
- <FIELD 1>
- <FIELD 2>
- <FIELD 3>
- <FIELD 4>
- <FIELD 5>
- <FIELD 6>
- <FIELD 7>
- <FIELD 8>
- <FIELD 9>
...
OCR Text:
{ocr}
Respond with JSON only:
""".strip()

    return {
        "file_name": example["file_name"],
        "prompt": prompt,
        "response": example["ground_truth_json"]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Prepare prompt/response dataset from OCR TXT files + ground-truth JSONs"
    )
    parser.add_argument(
        "--txt_folder", type=str, required=True,
        help="Path to folder containing OCR TXT files (space-separated words)"
    )
    parser.add_argument(
        "--json_folder", type=str, required=True,
        help="Path to folder containing ground-truth JSONs"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Where to save the resulting HuggingFace dataset (e.g. data/dataset.arrow)"
    )
    args = parser.parse_args()

    raw = load_raw_examples(args.txt_folder, args.json_folder)
    if len(raw) == 0:
        raise RuntimeError("No examples found. Check your folders.")

    # Build prompt/response pairs
    formatted = [create_prompt_response(ex) for ex in raw]

    # Build a Dataset from dict-lists
    ds = Dataset.from_dict({
        "file_name":    [f["file_name"] for f in formatted],
        "prompt":       [f["prompt"]    for f in formatted],
        "response":     [f["response"]  for f in formatted],
    })

    # Split into train/val/test (80/10/10)
    split1 = ds.train_test_split(test_size=0.2, seed=42)
    tv     = split1["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({
        "train":      split1["train"],
        "validation": tv["train"],
        "test":       tv["test"]
    })

    # Save to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset_dict.save_to_disk(args.output_path)
    print(f"Saved DatasetDict → {args.output_path}")

if __name__ == "__main__":
    main()
