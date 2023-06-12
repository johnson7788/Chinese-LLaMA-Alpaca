#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import tqdm
import peft
import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig,PeftModel
from itertools import compress
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
# 日志保存的路径，保存到当前目录下的logs文件夹中
log_path = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = os.path.join(log_path, "api.log")
# 日志的格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_response(text, history=[],with_prompt=True):
    """
    获取答案
    """
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.4,
        max_new_tokens=400
    )
    prompt_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )
    def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})
    with torch.no_grad():
        print("开始进行推理预测")
        if with_prompt is True:
            input_text = generate_prompt(instruction=text)
        else:
            input_text = text
        inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
        generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device),
            attention_mask = inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s,skip_special_tokens=True)
        if args.with_prompt:
            response = output.split("### Response:")[1].strip()
        else:
            response = output
        print(f"输入是: {text}\n")
        print(f"回复是: {response}\n")
    history = history + [text, response]
    return response,history


def load_model(args):
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    model.eval()
    return model, tokenizer, device

@app.route("/api/chat", methods=['POST'])
def chat():
    """
    Args: 基于aspect的情感分析，给定实体，判断实体在句子中的情感
    """
    jsonres = request.get_json()
    # 可以预测多条数据
    data = jsonres.get('data', None)
    if not data:
        return jsonify({"code": 400, "msg": "data不能为空"}), 400
    logging.info(f"数据分别是: {data}")
    input = data.get('text', '')
    history = data.get('history', [])
    response, history = get_response(text=input, history=history)
    result = {"response": response}
    logging.info(f"返回的结果是: {result}")
    return jsonify(result)

@app.route("/ping", methods=['GET', 'POST'])
def ping():
    """
    测试
    :return:
    :rtype:
    """
    return jsonify("Pong")

def parse_args():
    """
    返回arg变量和help
    :return:
    """
    parser = argparse.ArgumentParser(description="加载模型",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--load_in_8bit', action='store_true', help='use 8 bit model')
    return parser.parse_args(), parser.print_help

if __name__ == '__main__':
    args, helpmsg = parse_args()
    model, tokenizer,device = load_model(args)
    app.run(host='0.0.0.0', port=7058, debug=False, threaded=True)

