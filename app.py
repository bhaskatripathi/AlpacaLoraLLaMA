
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Answer step by step.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Answer step by step.

### Instruction:
{instruction}

### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())

import streamlit as st
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

model_name = 'bhaskar/LLaMA-7B-peft'
tokenizer = LLaMATokenizer.from_pretrained(model_name)
model = LLaMAForCausalLM.from_pretrained(model_name).cuda()
generation_config = GenerationConfig(
    do_sample=True,
    max_length=1024,
    top_p=0.9,
    temperature=1.0,
    no_repeat_ngram_size=3,
    num_return_sequences=1,
)

def generate_prompt(instruction):
    return f"### Instruction: {instruction}\n\n### Response:"

def evaluate1(instruction):
    prompt = generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

def main():
    st.set_page_config(page_title="LLaMA-7B Language Model")
    st.title("LLaMA-7B Language Model")
    st.write("This is a LLaMA-7B language model fine-tuned on various text datasets to generate text for a given task. It was trained on PyTorch by and is capable of generating high-quality, coherent text that is similar to human writing. The model is highly versatile and can be used for a variety of tasks, including text completion, summarization, and translation.")
    instruction = st.text_area("Instruction", height=200)
    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            output = evaluate1(instruction)
        st.write(output)

if __name__ == "__main__":
    main()
