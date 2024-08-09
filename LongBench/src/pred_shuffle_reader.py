# -*- coding: utf-8 -*-

import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from new_utils import replace_head,shorten, add_eod
from vllm import LLM, SamplingParams


# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, tokenized_prompt, prompt):
    # return tokenizer.build_prompt(prompt)
    ids=tokenizer('<human>: \n <bot>:')
    l=len(ids.input_ids)
    if len(tokenized_prompt)>max_length:
        return '<s><human>:'+prompt[:-l] +'\n<bot>:'
    else:
        return '<s><human>:'+prompt+'\n<bot>:'
    
def build_chat_baichuan(tokenizer, tokenized_prompt, prompt):
    prompt='<reserved_102>'+prompt+'<reserved_103>'
    return prompt


def get_pred(model,tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_path):
    preds = []
    prompts = []
    answers = []
    all_classes = []
    questions = []
    outputs=[]

    for json_obj in tqdm(data):
        context=shorten(add_eod(json_obj['context']))
        context = shorten(context)
        json_obj['context']=context
        prompt = prompt_format.format(**json_obj)
        # prompt=prompt.replace('<eod>','\n\n')
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset : #not in ["lcc", "repobench-p", "trec", "nq", "triviaqa", "lsht"]: # chat models are better off without build prompt on these tasks
            if 'Baichuan' in model_path or 'baichuan' in model_path:
                prompt = build_chat_baichuan(tokenizer, tokenized_prompt, prompt)
            else:
                prompt=build_chat(tokenizer,tokenized_prompt,prompt)
        prompts.append(prompt)
        answers.append(json_obj['answers'])
        all_classes.append(json_obj['all_classes'])
        questions.append(json_obj['input'])

    
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512,stop=tokenizer.eos_token)
    # # Create an LLM.
    # llm = LLM(model=model_path,
    #         trust_remote_code=True,
    #         tensor_parallel_size=2)
    
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
        outputs.append(pred)

    
    # outputs = llm.generate(prompts, sampling_params)
    for input, pred, answer, all_classe, question in zip(prompts, outputs, answers, all_classes, questions):
        preds.append({"pred": pred.split('我的答案是')[-1], "answers": answer, "all_classes": all_classe, "input": input, "question": question})

    return preds


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser('test models on long doc QA')
    parser.add_argument('--model_path',default='/cognitive_comp/hejunqing/projects/chatGPT/checkpoints/llama-13b-sft-searchv2-flash8_fix/global_step3400-bf16',type=str)
    parser.add_argument('--out_file',default='step3500_bf16',type=str)
    parser.add_argument('--out_dir',default='pred_searchvingv5_rerun_1112_shuffle',type=str)

    args=parser.parse_args()
    datasets = ["dureader"]#"hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", \
        #"qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model (ChatGLM2-6B, for instance)
    if 'chatglm2' in args.model_path:
        model = AutoModel.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)#("THUDM/chatglm2-6b", trust_remote_code=True)·
    if 'Baichuan' in args.model_path:
        max_length = 3500
    elif 'v1.1' in args.model_path:
        max_length= 2048-512
    else:
        max_length=7500

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r")) ## dureader,"请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    # predict on each dataset
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    dataset = "dureader"
    data = []
    with open("shuffled_longbench.json", "r") as f:
        for line in f:
            data.append(json.loads(line))
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    preds = get_pred(model,tokenizer, data, max_length, max_gen, prompt_format, dataset, device, args.model_path)
    for pred in preds:
        print(pred)
    with open(f"{args.out_dir}/{args.out_file}-{dataset}.jsonl", "w") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')

    print('finish inference')
