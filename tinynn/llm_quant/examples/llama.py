import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from tinynn.llm_quant.modules import quant_fc


def basic_usage(model_path='huggyllama/llama-7b', quant_mod='dynamic'):
    device = torch.device('cuda')

    # load LLM model from huggingface or local path
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Do quantization.
    if quant_mod != 'fp16':
        # If your LLM model is Llama-family, you can set fuse_qkv to fuse qkv linear and scaled-dot-product-attention.
        quant_fc(model, quant_mod=quant_mod, fuse_qkv=True)
    model.to(device)

    prompt = "Building a website can be done in 10 simple steps:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        top_k=1,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        use_cache=True,
    )

    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for output in outputs:
        print(output)


if __name__ == '__main__':
    basic_usage()
