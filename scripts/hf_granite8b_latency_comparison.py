import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compile", help = "Weather to run in compile mode or eager mode", action='store_true')
args = parser.parse_args()
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.2-8b-instruct", device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.2-8b-instruct")

#try out different modes and make a table
batch_texts = [
    "Hello, how are you?",
    "How old are you?",
    "What is the cooperative game theoretic explanation for SHAP values?",
    "What did John Paul Sarte argue about existentialism?"
]
#include warmup for torch compile and compute latency for other batches not including the warmup
#include torch export to illustrate the increase in latency
first_token_latencies = []
next_token_latencies = []
generated_texts = []
if args.compile:
    start_compile = time.time()
    # model.compile(mode="max-autotune-no-cudagraphs")
    model.compile()
    end_compile = time.time()
    compile_time = end_compile - start_compile
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
                    start_time = time.time()
                    first_token_output = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    end_time = time.time()
                    first_token_latency = end_time - start_time
                    first_token_latencies.append(first_token_latency)                
                    start_time = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    generated_length = outputs.shape[1] - inputs['input_ids'].shape[1]
                    if generated_length > 1:
                        remaining_time = total_time - first_token_latency
                        avg_next_token_time = remaining_time / (generated_length - 1)
                        next_token_latencies.append(avg_next_token_time)
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(generated_text)

    avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0
    avg_next_token_latency = sum(next_token_latencies) / len(next_token_latencies) if next_token_latencies else 0

    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Total compile time: {compile_time:.4f} seconds")

else:
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
                    start_time = time.time()
                    first_token_output = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    end_time = time.time()
                    first_token_latency = end_time - start_time
                    first_token_latencies.append(first_token_latency)                
                    start_time = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    generated_length = outputs.shape[1] - inputs['input_ids'].shape[1]
                    if generated_length > 1:
                        remaining_time = total_time - first_token_latency
                        avg_next_token_time = remaining_time / (generated_length - 1)
                        next_token_latencies.append(avg_next_token_time)
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(generated_text)

    avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0
    avg_next_token_latency = sum(next_token_latencies) / len(next_token_latencies) if next_token_latencies else 0
  

print(f"Average first token latency: {avg_first_token_latency:.4f} seconds")
print(f"Average next token latency: {avg_next_token_latency:.4f} seconds")
print(f"{'='*60}")
