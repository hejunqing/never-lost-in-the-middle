import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM,LlamaForCausalLM
from tqdm import tqdm
from new_utils import replace_head,shorten,add_eod


# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, tokenized_prompt, prompt):
    # return tokenizer.build_prompt(prompt)
    
    return '<s><human>:'+prompt+'\n<bot>:'


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device):
    preds = []
    count=0
    for json_obj in tqdm(data):
        if dataset=='dureader' or dataset=='passage_retrieval_zh' or dataset=='vcsum':
            context=add_eod(replace_head(json_obj['context'],dataset))
            # if len(context)>max_length-50:
            #     context=context[:max_length-50]
            json_obj['context']=context#shorten(context,num=10)
        elif dataset=='multifieldqa_zh':
            context=add_eod(json_obj['context'],dataset)
            # if len(context)>max_length:
            #     context=context[:max_length-50]
            json_obj['context']=context

        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        ids=tokenizer('<s><human>: \n <bot>:')
        l=len(ids.input_ids)
        if len(tokenized_prompt) > max_length-l:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset : #not in ["lcc", "repobench-p", "trec", "nq", "triviaqa", "lsht"]: # chat models are better off without build prompt on these tasks
            prompt = build_chat(tokenizer, tokenized_prompt, prompt)
        input_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=False).to(device).input_ids
        context_length = input_ids.shape[-1]
        output = model.generate(
            input_ids,
            max_new_tokens=max_gen,
            do_sample=True,
            temperature=0.8,
            top_p=0.85,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,repetition_penalty=1.,
        )[0]
        pred = tokenizer.decode(output, skip_special_tokens=False)
        pred = pred.split('\n<bot> : ')[-1].rstrip('</s>')
        # pred = pred.split("我的答案是")[-1]
        if count<5:
            print('input:',prompt)
            print('answer:',pred)
        count+=1
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],"input":prompt,"question":json_obj['input']})
    return preds


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser('test models on long doc QA')
    parser.add_argument('--model_path',default='/cognitive_comp/hejunqing/projects/chatGPT/checkpoints/llama-13b-sft-searchv2-flash8_fix/global_step3400-bf16',type=str)
    parser.add_argument('--out_file',default='step3500_bf16',type=str)
    parser.add_argument('--out_dir',default='pred_searchingv5_all_zh_1010_latest',type=str)

    args=parser.parse_args()
    datasets = ["dureader","passage_retrieval_zh","vcsum"]##"dureader","hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", \
        #"qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model (ChatGLM2-6B, for instance)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)#("THUDM/chatglm2-6b", trust_remote_code=True)
    if 'chatglm2' in args.model_path:
        model = AutoModel.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    # define max_length
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r")) ## dureader,"请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    
    # predict on each dataset
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for dataset in datasets:
        data = load_dataset('data/LongBench.py', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        max_length = 8192-max_gen-100
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device)
        with open(f"{args.out_dir}/{args.out_file}-{dataset}.jsonl", "w",encoding="utf8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
    print('finish inference')
