# python src/infer.py \
#   --checkpoint_dir models/llama1b-lora \
#   --txt_folder new_invoices/ocr_txt \
#   --output_folder outputs/jsons

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_infer_prompt(ocr_text: str) -> str:
    return f"""
Given the OCR text below, extract the following details in well-formatted JSON exactly as shown, ensuring all "N/A" values remain unchanged.
Extract and include the following fields (Placeholder names, change them accordingly):
- <FIELD 1>
- <FIELD 2>
- <FIELD 3>
- <FIELD 4>
- <FIELD 5>
- <FIELD 6>
- <FIELD 7>
- <FIELD 8>
- <FIELD 9>
.
.
.
.

OCR Text:
{ocr_text}
Respond with JSON only:
""".strip()

def load_model_and_tokenizer(checkpoint_dir: str, device: torch.device):
    """
    Load both the tokenizer and the fine-tuned model from checkpoint_dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval().to(device)
    return tokenizer, model

def infer_one_txt(txt_path: str, tokenizer, model, device) -> str:
    # Read the .txt file (space-separated OCR tokens)
    with open(txt_path, "r", encoding="utf-8") as f:
        ocr_text = f.read().strip()

    prompt = create_infer_prompt(ocr_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False
        )
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Return everything from the first '{' onwards (JSON starts here)
    idx = decoded.find("{")
    return decoded[idx:].strip() if idx != -1 else decoded.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Batch‐infer invoice JSON from OCR .txt files using a fine‐tuned LLaMA+LoRA model"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to LoRA‐fine‐tuned checkpoint directory"
    )
    parser.add_argument(
        "--txt_folder", type=str, required=True,
        help="Folder of new OCR .txt files to process (space-separated tokens)"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Where to write the generated JSONs (one .json per .txt)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(args.checkpoint_dir, device)

    os.makedirs(args.output_folder, exist_ok=True)
    for fn in os.listdir(args.txt_folder):
        if not fn.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fn)[0]
        out_path = os.path.join(args.output_folder, base + ".json")
        if os.path.exists(out_path):
            print(f"Skipping {base}: already exists.")
            continue

        txt_path = os.path.join(args.txt_folder, fn)
        try:
            gen = infer_one_txt(txt_path, tokenizer, model, device)
            with open(out_path, "w", encoding="utf-8") as fo:
                fo.write(gen)
            print(f"Saved {base}.json")
        except Exception as e:
            print(f"[Error] {base}: {e}")

if __name__ == "__main__":
    main()
