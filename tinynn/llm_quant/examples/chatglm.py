# This script is based on https://github.com/THUDM/ChatGLM-6B
import signal
import os
import torch
from transformers import AutoModel, AutoTokenizer

from tinynn.llm_quant.modules import quant_fc


def basic_usage(model_path='THUDM/chatglm-6b', quant_mod='dynamic'):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device('cuda')

    # Do quantization.
    if quant_mod != 'fp16':
        quant_fc(model, quant_mod=quant_mod)
    model.to(device)

    clear_command = 'clear'
    stop_stream = False

    def build_prompt(history):
        prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
        for query, response in history:
            prompt += f"\n\n用户：{query}"
            prompt += f"\n\nChatGLM-6B：{response}"
        return prompt

    def signal_handler(signal, frame):
        global stop_stream
        stop_stream = True

    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == '__main__':
    basic_usage()
