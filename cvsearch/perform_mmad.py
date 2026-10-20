"""MMAD (MVTec-AD subset) multiple-choice evaluation adapter.

Reads the flattened per-question records of datasets/mmad_subset/splits/<split>_questions.jsonl and answers each
question with the VLM.

Modes:
  --direct-answer                 vanilla backbone baseline (no visual search)
  --search-target {object,defect} CVSearch pipeline; MMAD questions name no object, so the SAM3/search target is fixed:
                                  object = the product class (screw/capsule/leather), defect = the word "defect"
"""
import argparse
import collections
import json
import os
import re
import sys
import warnings

import torch
from PIL import Image
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
from models.modeling_qwenvl import ModelQwenVL

warnings.filterwarnings("ignore")


def format_question(question, options):
    ret = question
    for letter, text in options.items():
        ret += f"\n{letter}. {text}"
    letters = list(options)
    letter_str = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    ret += (f"\nSelect the best answer to the above multiple-choice question based on the image. "
            f"Respond with only the letter ({letter_str}) of the correct option.\nThe best answer is:")
    return ret


def parse_letter(text, letters):
    m = re.match(r"\s*\(?([A-Za-z])\b", text)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m and m.group(1) in letters else None


def anomaly_detection_metrics(records, options_by_qid):
    """Precision/recall for the Anomaly Detection question. Positive = anomalous image.

    The Yes/No option order is shuffled per question, so the predicted letter is mapped to its option text.
    """
    def prf(rs):
        tp = sum(a and p for a, p in rs)
        fn = sum(a and not p for a, p in rs)
        fp = sum((not a) and p for a, p in rs)
        tn = sum((not a) and not p for a, p in rs)
        div = lambda x, y: round(100 * x / y, 2) if y else None
        prec, rec = div(tp, tp + fp), div(tp, tp + fn)
        f1 = round(2 * prec * rec / (prec + rec), 2) if prec and rec else None
        return {"n": len(rs), "tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": prec, "recall": rec,
                "f1": f1, "specificity": div(tn, tn + fp)}

    rows = []
    for r in records:
        if r["question_type"] != "Anomaly Detection" or r["pred"] is None:
            continue
        text = options_by_qid[r["question_id"]][r["pred"]].strip(". ").lower()
        rows.append((r["category"], bool(r["is_anomaly"]), text == "yes"))
    out = {"overall": prf([(a, p) for _, a, p in rows])}
    out["by_category"] = {c: prf([(a, p) for cc, a, p in rows if cc == c]) for c in sorted({c for c, _, _ in rows})}
    return out


def summarize(records, options_by_qid=None):
    def acc(rs):
        return {"n": len(rs), "correct": sum(r["correct"] for r in rs),
                "acc": round(100 * sum(r["correct"] for r in rs) / len(rs), 2) if rs else None}

    out = {"overall": acc(records), "unparsed": sum(r["pred"] is None for r in records)}
    for key in ["question_type", "category", "is_anomaly", "defect_type"]:
        groups = collections.defaultdict(list)
        for r in records:
            groups[str(r[key])].append(r)
        out[f"by_{key}"] = {k: acc(v) for k, v in sorted(groups.items())}
    if options_by_qid is not None:
        out["anomaly_detection"] = anomaly_detection_metrics(records, options_by_qid)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data-path", type=str, default="datasets/mmad_subset")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "val", "test"])
    parser.add_argument("--vlm-device", type=str, default="cuda:0")
    parser.add_argument("--direct-answer", action="store_true", help="vanilla backbone baseline (only mode implemented)")
    parser.add_argument("--search-target", type=str, default=None, choices=["object", "defect"],
                        help="run CVSearch with a fixed search target")
    parser.add_argument("--force-tree", action="store_true",
                        help="ignore SAM3 detections: always use the tree search (SAM3 image features still build the tree)")
    parser.add_argument("--no-second-search", action="store_true",
                        help="do not re-run SAM3 on the best tree candidate (search_mode=False)")
    parser.add_argument("--sam-model-path", type=str, default="models/sam3/sam3/sam3.pt")
    parser.add_argument("--nlp-model-path", type=str, default="models/en_core_web_sm-3.8.0")
    parser.add_argument("--sam-device", type=str, default="cuda:1")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (debug)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="recompute summary.json from an existing outputs.jsonl (no model is loaded)")
    args = parser.parse_args()

    if not args.direct_answer and not args.search_target and not args.rescore:
        raise SystemExit("choose a mode: --direct-answer or --search-target {object,defect}")
    mode = "direct_answer" if args.direct_answer else f"cvsearch_{args.search_target}" + ("_tree" if args.force_tree else "")

    data_root = os.path.join(args.root_path, args.data_path)
    model_path = os.path.join(args.root_path, args.model_path)
    questions = [json.loads(l) for l in open(os.path.join(data_root, "splits", f"{args.split}_questions.jsonl"))]
    if args.limit:
        questions = questions[:args.limit]

    out_dir = args.output_dir or os.path.join(
        args.root_path, "results", f"mmad_{os.path.basename(args.model_path)}_{mode}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    options_by_qid = {q["question_id"]: q["options"] for q in questions}
    outputs_path = os.path.join(out_dir, "outputs.jsonl")
    if args.rescore:
        records = [json.loads(l) for l in open(outputs_path)]
    else:
        model = ModelQwenVL(model_path=model_path, device=args.vlm_device, torch_dtype=torch.bfloat16, patch_scale=1.2)
        if args.search_target:
            import spacy
            from models.modeling_sam3 import sam3_inference
            from CVSearch import get_cvsearch_response
            nlp = spacy.load(name=os.path.join(args.root_path, args.nlp_model_path))
            sam3 = sam3_inference(model_path=os.path.join(args.root_path, args.sam_model_path), device=args.sam_device)
            # same search hyper-parameters as the Qwen branch of perform_CVSearch.py
            search_kwargs = {
                "pop_limit": lambda max_depth: max_depth * 3,
                "threshold_descrease": [0.05, 0.1, 0.2],
                "answering_confidence_threshold_lower": 0,
                "answering_confidence_threshold_upper": 0.9,
                "fast_threshold": 0.6,
            }
            ic_examples = json.load(open("ic_examples/vstar.json"))
        records = []
        with open(outputs_path, "w") as f:
            for q in tqdm(questions):
                image_file = os.path.join(data_root, q["image_path"])
                prompt = format_question(q["question"], q["options"])
                extra = {}
                if args.search_target:
                    ann = {"input_image": image_file, "question": q["question"], "answer_type": "preformatted",
                           "final_prompt": prompt}
                    fixed = [q["category"]] if args.search_target == "object" else ["defect"]
                    raw = get_cvsearch_response(
                        sam_model=sam3, zoom_model=model, nlp_model=nlp, annotation=ann, ic_examples=ic_examples,
                        decomposed_question_template="What is the appearance of the {}?", fixed_targets=fixed,
                        force_tree=args.force_tree, search_mode=not args.no_second_search, **search_kwargs)
                    extra = {k: ann.get(k) for k in ["search_mode", "root_ans_conf", "targets", "sam", "num_pop",
                                                     "searched_bbox"]}
                else:
                    image = Image.open(image_file).convert("RGB")
                    raw = model.free_form_using_nodes(image, prompt, [])  # empty node list = direct answering
                pred = parse_letter(raw, list(q["options"]))
                rec = {k: q[k] for k in ["question_id", "image_id", "category", "question_type", "is_anomaly",
                                         "defect_type", "answer"]}
                rec.update(pred=pred, correct=pred == q["answer"], raw_output=raw, **extra)
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize(records, options_by_qid)
    summary["config"] = {"model": args.model_path, "split": args.split, "mode": mode}
    if any("search_mode" in r for r in records):
        summary["search_mode_counts"] = dict(collections.Counter(str(r.get("search_mode")) for r in records))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

    print("\n" + "=" * 60)
    print(f"Overall: {summary['overall']['acc']}% ({summary['overall']['correct']}/{summary['overall']['n']}), "
          f"unparsed: {summary['unparsed']}")
    for key in ["by_question_type", "by_category", "by_is_anomaly"]:
        print(f"\n[{key}]")
        for k, v in summary[key].items():
            print(f"  {k:<26}{v['acc']:>7}%  ({v['correct']}/{v['n']})")
    ad = summary["anomaly_detection"]
    print("\n[anomaly_detection] positive = anomalous image")
    for name, m in [("overall", ad["overall"])] + list(ad["by_category"].items()):
        print(f"  {name:<10} P={m['precision']}%  R={m['recall']}%  F1={m['f1']}%  spec={m['specificity']}%  "
              f"(TP {m['tp']} FN {m['fn']} FP {m['fp']} TN {m['tn']})")
