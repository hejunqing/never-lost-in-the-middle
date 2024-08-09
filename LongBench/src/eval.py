import os
import json

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "nq": qa_f1_score,
    "triviaqa": qa_f1_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench_p": code_sim_score,
}

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def process_search_pred(line):
    if '<human>' in line['pred'] and '<bot>' in line['pred']:
        pred=line['pred'].split('<bot>')[-1]
        pred=pred.strip(':')

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser('test models on long doc QA')
    scores = dict()
    
    parser.add_argument('--out_file',default='',type=str)
    parser.add_argument('--out_dir',default='pred_gpt3.5-16k_shuffle',type=str)

    args=parser.parse_args()
    all_files = os.listdir(args.out_dir)
    for filename in all_files:
        predictions, answers = [], []
        dataset = filename.split('.')[0]
        dataset = dataset.split('-')[-1]
        with open(f"{args.out_dir}/{filename}", "r") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"].split('我的答案是')[-1])
                # predictions.append()
                answers.append(data["answers"])
                all_classes = data["all_classes"] if "all_classes" in data else None
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    with open(f"{args.out_file}-result.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
