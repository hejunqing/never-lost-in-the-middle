import os,sys
sys.path.append(os.path.abspath('../LongBench'))


from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM,LlamaForCausalLM
from tqdm import tqdm
from new_utils import replace_head,shorten,add_eod
from data.set_position import load_json


# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, tokenized_prompt, prompt):
    # return tokenizer.build_prompt(prompt)
    
    return '<s><human>:'+prompt+'\n<bot>:'

def build_chat2(tokenizer, tokenized_prompt,prompt):
    return tokenizer.build_prompt(prompt)

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device,model_path):
    preds = []
    prompts = []
    answers = []
    all_classes = []
    questions = []

    count=0
    for json_obj in tqdm(data):
        if dataset=='dureader' or dataset=='passage_retrieval_zh' or dataset=='vcsum':
            # context=add_eod(replace_head(json_obj['context'],dataset))
            if 'reader' in model_path or 'search' in model_path:
                context=add_eod(json_obj['context'].replace('\n\n','。'),dataset)
            # if len(context)>max_length-50:
            #     context=context[:max_length-50]
            else: context=json_obj['context']
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
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2) - 15
            print("11111111")
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if 'chatglm' in  model_path: #not in ["lcc", "repobench-p", "trec", "nq", "triviaqa", "lsht"]: # chat models are better off without build prompt on these tasks
        #     prompt = build_chat2(tokenizer, tokenized_prompt, prompt)
        # else:    
        #     prompt = build_chat(tokenizer, tokenized_prompt, prompt)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]        
        assert len(tokenized_prompt) <= 8192
        prompts.append(prompt)
        answers.append(json_obj['answers'])
        all_classes.append(json_obj['all_classes'])
        questions.append(json_obj['input'])


        # input_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=False).to(device).input_ids
        # context_length = input_ids.shape[-1]
        # output = model.generate(
        #     input_ids,
        #     max_new_tokens=max_gen,
        #     do_sample=True,
        #     temperature=0.8,
        #     top_p=0.85,
        #     eos_token_id=tokenizer.eos_token_id,
        #     bos_token_id=tokenizer.bos_token_id,
        #     pad_token_id=tokenizer.pad_token_id,
        #     early_stopping=True,repetition_penalty=1.,
        # )[0]
        # if 'reader' in model_path or 'search' in model_path:
        #     pred= tokenizer.decode(output, skip_special_tokens=False)
        #     pred = pred.split('\n<bot> : ')[-1]#.rstrip('</s>')
        # else:
        #     pred = tokenizer.decode(output[context_length:], skip_special_tokens=False)
        
        # # pred = pred.split("我的答案是")[-1]
        # if count<5:
        #     print('input:',prompt)
        #     print('answer:',pred)
        # count+=1
        # preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],"input":prompt,"question":json_obj['input']})
    outputs = []
    for i, prompt in enumerate(prompts):
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = model.chat(tokenizer, messages)
    
        outputs.append(response)
        print("======={}========".format(i))
        print("prompt2",prompts[i])
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0] 
        print(len(tokenized_prompt))
        print("dataset", response)
        print("dataset_split",response.split("我的答案是")[-1])
        # print("dataset_len", len(tokenizer(output.outputs[0].text, truncation=False, return_tensors="pt").input_ids[0]))
    for input, pred, answer, all_classe, question in zip(prompts, outputs, answers, all_classes, questions):
        preds.append({"pred": pred, "answers": answer, "all_classes": all_classe, "input": input, "question": question})

    return preds






def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser('test models on long doc QA')
    # parser.add_argument('--model_path',default='/cognitive_comp/hejunqing/projects/chatGPT/checkpoints/llama-13b-searchv5-flash8/global_step3500-bf16',type=str)
    # parser.add_argument('--test_file',default='/cognitive_comp/hejunqing/projects/LongBench/data/Lost_retrieval/passage_retrieval_1.json',type=str)
    # parser.add_argument('--out_dir',default='pred_ziya_reader_single_greedy',type=str)


    parser.add_argument('--model_path',default='/cognitive_comp/liuyibo/cg/pretrain_model/Baichuan2-13B-Chat',type=str)

    parser.add_argument('--test_file',default='/cognitive_comp/liuyibo/cg/LongBench/data/Lost_retrieval/passage_retrieval_zh_20.json',type=str)
    parser.add_argument('--out_dir',default='pred_Baichuan2-13B-Chat_retrieval_single',type=str)




    args=parser.parse_args()
    seed_everything(42)
    datasets = ["passage_retrieval_zh"]##"dureader","hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", \
        #"qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model (ChatGLM2-6B, for instance)
    from transformers.generation.utils import GenerationConfig

    if 'chatglm' in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)#
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,revision="v2.0",
        use_fast=False, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                    revision="v2.0",
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True).to(device)

        model.generation_config = GenerationConfig.from_pretrained(args.model_path, revision="v2.0")
        
        model.generation_config.temperature = 0.2
        model.generation_config.max_new_tokens = 100

    model = model.eval()
    # define max_length
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r")) ## dureader,"请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    
    # predict on each dataset
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for dataset in datasets:
        # data = load_dataset('data/LongBench.py', dataset, split='test')
        data=load_json(args.test_file)
        tag=os.path.basename(args.test_file).strip('.json')
        prompt_format = dataset2prompt["passage_retrieval_zh"]
        max_gen = dataset2maxlen["passage_retrieval_zh"]
        if 'chatglm' in args.model_path:
            max_length = 32000-max_gen-100
        else:
            # max_length=8192-max_gen-100
            max_length=4096-max_gen-100
        print("max_length", max_length)
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device,args.model_path)
        with open(f"{args.out_dir}/{tag}.jsonl", "w",encoding="utf8") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False))
                f.write('\n')
    print('finish inference')
