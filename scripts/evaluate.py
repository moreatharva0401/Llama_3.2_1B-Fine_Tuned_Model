# python scripts/evaluate.py \
#   --pred_folder outputs/jsons \
#   --gold_folder data/ground_truth_json \
#   --field_metrics_json field_accuracy_with_confusion_and_f1.json \
#   --out_report outputs/eval_report.json


import os
import json
import argparse
from glob import glob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_json_str(j: dict) -> str:
    """
    Serialize to compact, sorted keys so that BLEU is not impacted by whitespace or key-order.
    """
    return json.dumps(j, sort_keys=True, separators=(","))

def bleu_score_single(reference: str, candidate: str) -> float:
    """
    Compute BLEU score (0–100) for one JSON string vs. its ground truth.
    We convert to token lists by splitting on whitespace.
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(
        [ref_tokens],
        cand_tokens,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    return score * 100.0

def compute_file_level_accuracy(pred_folder: str, gold_folder: str):
    all_bleus = []
    skipped = []
    for pred_path in glob(os.path.join(pred_folder, "*.json")):
        base = os.path.basename(pred_path)
        gold_path = os.path.join(gold_folder, base)
        if not os.path.exists(gold_path):
            skipped.append(base)
            continue
        pred_json = load_json(pred_path)
        gold_json = load_json(gold_path)

        ref_str = normalize_json_str(gold_json)
        cand_str = normalize_json_str(pred_json)
        score = bleu_score_single(ref_str, cand_str)
        all_bleus.append((base, score))

    return all_bleus, skipped

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated JSONs against ground-truth JSONs"
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="Folder containing your model's generated JSONs",
    )
    parser.add_argument(
        "--gold_folder",
        type=str,
        required=True,
        help="Folder containing the ground-truth JSONs",
    )
    parser.add_argument(
        "--field_metrics_json",
        type=str,
        required=False,
        help="(Optional) precomputed field-level metrics JSON",
    )
    parser.add_argument(
        "--out_report",
        type=str,
        default="evaluation_report.json",
        help="Where to write per-file BLEU scores + summary",
    )

    args = parser.parse_args()

    #File-level BLEU
    bleus, skipped = compute_file_level_accuracy(args.pred_folder, args.gold_folder)
    if len(bleus) == 0:
        print("No overlapping JSONs to score. Exiting.")
        return

    #Save per-file scores
    file_scores = {fname: score for (fname, score) in bleus}
    avg_bleu = sum(score for _, score in bleus) / len(bleus)

    report = {
        "file_count_evaluated": len(bleus),
        "average_file_bleu": avg_bleu,
        "skipped_due_to_missing_gold": skipped,
        "per_file_bleu": file_scores,
    }

    #If field_metrics_json is provided, merge it in
    if args.field_metrics_json and os.path.exists(args.field_metrics_json):
        with open(args.field_metrics_json, "r", encoding="utf-8") as f:
            field_metrics = json.load(f)
        report["field_level_metrics"] = field_metrics

    #Dump to out_report
    with open(args.out_report, "w", encoding="utf-8") as fo:
        json.dump(report, fo, indent=2)

    print(f"Saved evaluation report → {args.out_report}")
    print(f"Avg BLEU over {len(bleus)} files: {avg_bleu:.2f}")

if __name__ == "__main__":
    main()
