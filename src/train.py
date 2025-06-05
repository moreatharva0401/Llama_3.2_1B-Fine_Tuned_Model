# python src/train.py \
#   --dataset_path data/dataset.arrow \
#   --output_dir models/llama1b-lora \
#   --epochs 3 \
#   --batch_size 1

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

def tokenize_fn(example, tokenizer):
    full = example["prompt"] + "\n" + example["response"] + tokenizer.eos_token
    tok = tokenizer(full, padding="max_length", truncation=True, max_length=2048)
    labels = tok["input_ids"].copy()

    # Determine how many tokens belong to the prompt alone
    prompt_len = len(tokenizer(example["prompt"], truncation=True)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    tok["labels"] = labels
    return tok

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 1B with LoRA adapters")
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the HuggingFace dataset directory (arrow format), e.g. data/dataset.arrow"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the LoRA‐adapted model checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per‐device batch size"
    )
    args = parser.parse_args()

    # Load DatasetDict
    ds = load_from_disk(args.dataset_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all splits
    tokenized = ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["file_name", "prompt", "response"],
        batched=False
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load base LLaMA 1B model
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Configure LoRA
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=2e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        warmup_steps=5,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=3407,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Final model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
