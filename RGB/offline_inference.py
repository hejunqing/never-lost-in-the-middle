#encoding=utf8

import json
import torch
import random
import argparse
import xlsxwriter
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.conversation import get_conv_template


def init_llm(model_path, max_length=4096):
    # Create an LLM.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        swap_space=16,
        dtype="bfloat16",
    )
    return llm

def baichuan_prompt(query):
    conv = get_conv_template("baichuan-chat")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def chatglm2_prompt(query):
    conv = get_conv_template("chatglm2")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def qwen_prompt(query):
    conv = get_conv_template("qwen-7b-chat")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def intern_prompt(query):
    conv = get_conv_template("internlm-chat")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def ziya_prompt(query):
    conv = get_conv_template("ziya_13b")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

if __name__ == '__main__':
    # arguments containing: config_path, ckpt_path, max_length
    args_parser = argparse.ArgumentParser("vllm offline inference")
    args_parser.add_argument("--input_path", type=str, help="input path", required=True)
    args_parser.add_argument("--output_path", type=str, help="output path", required=True)
    args_parser.add_argument("--model_path", type=str, help="model path", required=True)
    args_parser.add_argument("--type", type=str, choices=["baichuan", "chatglm", "qwen", "intern", "ziya"], help="type", default="ziya")
    args_parser.add_argument("--beam", type=int, help="num_answers", default=1)
    args_parser.add_argument("--max_length", type=int, help="max_length", default=4096)
    args_parser.add_argument("--sbs", action="store_true", default=False)
    args_parser.add_argument("--ife", action="store_true", default=False, help="instruction following eval")
    args = args_parser.parse_args()
    print("args", args)

    if args.sbs or args.ife:
        data_list = []
        for line in open(args.input_path):
            data = json.loads(line.strip())
            data_list.append(data)
    else:
        data_list = json.load(open(args.input_path))
        random.shuffle(data_list)
    print("data size:", len(data_list))

    llm = init_llm(args.model_path, args.max_length)

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.85,
        top_k=-1,
        max_tokens=args.max_length,
        n=args.beam,
    )

    prompts = []
    queries = []
    for data in tqdm(data_list):
        if args.sbs:
            query = data["text"]
        elif args.ife:
            query = data["prompt"]
        else:
            query = data["conversations"][0]["value"]
        if args.type == "baichuan":
            prompt = baichuan_prompt(query)
        elif args.type == "chatglm2":
            prompt = chatglm2_prompt(query)
        elif args.type == "ziya":
            prompt = ziya_prompt(query)
        elif args.type == "qwen":
            prompt = qwen_prompt(query)
        elif args.type == "intern":
            prompt = intern_prompt(query)
        prompts.append(prompt)
        queries.append(query)
        
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    
    # outputs = []
    # for prompt in tqdm(prompts):
    #     try:
    #         output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    #         outputs.append(output)
    #     except:
    #         print(f"prompt generating failed: {prompt}")
    #         continue
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Print the outputs.
    if args.sbs:
        workbook = xlsxwriter.Workbook(args.output_path)
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "class")
        worksheet.write(0, 1, "query")
        worksheet.write(0, 2, "answer")
        for idx, (query, output) in enumerate(zip(queries, outputs)):
            worksheet.write(idx + 1, 0, "写作")
            worksheet.write(idx + 1, 1, query)
            worksheet.write(idx + 1, 2, output.outputs[0].text)
        workbook.close()
    elif args.ife:
        fout = open(args.output_path, mode="w")
        for query, output in zip(queries, outputs):
            response = output.outputs[0].text
            output_data = {"prompt": query, "response": response}
            # print(output_data)
            print(json.dumps(output_data, ensure_ascii=False), file=fout)
            fout.flush()
        fout.close()
    else:
        fout = open(args.output_path, mode="w")
        for query, output in zip(queries, outputs):
            prompt = output.prompt
            generated_texts = [c_output.text for c_output in output.outputs]
            output_data = {"prompt": query, "type": args.type, "answers": generated_texts}
            print(output_data)
            print(json.dumps(output_data, ensure_ascii=False), file=fout)
            fout.flush()
        fout.close()