import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compile", help = "Weather to run in compile mode or eager mode", action='store_true')
args = parser.parse_args()
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.2-8b-instruct", device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.2-8b-instruct")

#reduce-overhead doesn't work here

batch_texts = [
    "Hello, how are you?",
    "How old are you?",
    "What is the cooperative game theoretic explanation for SHAP values?",
    "What did John Paul Sarte argue about existentialism?"
]


model.compile()
generated_texts = []
for text in batch_texts:
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                torch.cuda.synchronize() # try with and without it
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)


# print("\nGENERATED TEXTS:")
# for i, (input_text, generated_text) in enumerate(zip(batch_texts, generated_texts)):
#     print(f"Input {i+1}: {input_text}")
#     print(f"Output {i+1}: {generated_text}")
#     print("-" * 50)