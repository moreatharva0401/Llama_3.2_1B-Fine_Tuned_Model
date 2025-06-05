Invoice Data Extraction with Fine-Tuned LLaMA


Overview

This repository contains code and resources for a Generative AI-based invoice data extraction system, fine-tuned from LLaMA using LoRA adapters to perform structured information extraction (JSON) from raw OCR CSVs. Trained and evaluated in Google Colab, the final model achieves over 90% invoice-level accuracy and field-level F1 scores above 0.90 on key fields.


Features

- Prompt-based extraction of invoice fields (invoice_id, invoice_date, totals, customer/vendor info, line items, etc.)
- LoRA-optimized fine-tuning of LLaMA 1B for reduced GPU footprint (~7GB VRAM)
- End-to-end pipeline: OCR CSV -> Prompt generation -> Model inference -> JSON output
- Skip-existing check in inference for incremental batch processing
- Field-level and file-level accuracy evaluation metrics


Repository Structure
- data/        
   - ocr_csv/
   - ground_truth_json/
   - dataset.arrow            
- examples/                
- src/                     
   - prepare_data.py      
   - train.py             
   - infer.py             
- outputs/                 
- scripts/                 
   - evaluate.py
- requirements.txt         
- README.md                


Installation

- Clone this repository:
```bash
git clone https://github.com/<PMayekar18>/invoice-llama
cd invoice-llama
```

- Create and activate a Python 3.10+ virtual environment:
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate      # Windows
```

- Install dependencies:
```bash
pip install -r requirements.txt
```


Usage

1. Data Preparation

Place OCR .txt files in data/ocr_txt/
Place corresponding ground-truth JSONs (for training/eval) in data/ground_truth_json/
```bash
python src/prepare_data.py \
  --txt_folder data/ocr_txt \
  --json_folder data/ground_truth_json \
  --out_dataset data/dataset.arrow
```
This script:
- Reads each .txt as raw OCR words.
- Loads matching JSON (if user-supplied); otherwise you must create your own ground truth in JSON format.
- Builds prompt/response pairs with placeholder field names in the prompt.
- Splits into train/validation/test and saves a DatasetDict at data/dataset.arrow.

2. Fine-tuning
```bash
python src/train.py \
  --dataset_path data/dataset.arrow \
  --output_dir models/llama1b-lora \
  --epochs 3 \
  --batch_size 1
```
Key LoRA hyperparameters in train.py:
r: 16
alpha: 16
dropout: 0
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

3. Inference
```bash
python src/infer.py \
  --checkpoint models/llama1b-lora/checkpoint-XXXX \
  --txt_folder new_invoices/ocr_txt \
  --output_folder outputs/jsons
```

4. Evaluation
```bash
python scripts/evaluate.py \
  --pred_folder outputs/jsons \
  --gold_folder data/ground_truth_json \
  --field_metrics_json field_accuracy_with_confusion_and_f1.json \
  --out_report outputs/eval_report.json
```


Notes on Ground Truth

If you use your own OCR .txt data, you must supply matching ground‑truth JSON in data/ground_truth_json/ for training/evaluation. The keys in each JSON must follow the same structure shown in examples/.

The examples/ folder includes four sample invoice images, their OCR .txt outputs, and corresponding ground‑truth JSONs as templates.


Contact & License

This project was developed as part of an internship at Datamatics Global Services Ltd., TRUCap department. Feel free to open an issue or pull request.