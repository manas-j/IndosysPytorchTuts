import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compile", action='store_true')
args = parser.parse_args()

device = torch.device("cuda")

print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.2-8b-instruct", 
    device_map="auto",
    torch_dtype=torch.bfloat16 
)

print(f"Memory after model load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.2-8b-instruct")

torch.cuda.empty_cache()
print(f"Memory after cache clear: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

batch_texts = [
    "Hello, how are you?",
]

if args.compile:
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
                    with torch.profiler.profile(
                        profile_memory=True,
                        with_stack=True,
                        record_shapes=True,
                        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    torch.cuda.synchronize() # try with and without it
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(generated_text)
    filename = "granite8b_compiled_trace"
    prof.export_chrome_trace(filename+'.json')
    prof.export_memory_timeline(filename+'.html')
    print(f"Profiling data exported to {filename}.json")
    print(f"Memory profile exported to {filename}.html")

else:
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
                    with torch.profiler.profile(
                        profile_memory=True,
                        with_stack=True,
                        record_shapes=True,
                        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    torch.cuda.synchronize()
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(generated_text)
    filename = "granite8b_uncompiled_trace"
    prof.export_chrome_trace(filename+'.json')
    prof.export_memory_timeline(filename+'.html')
    print(f"Profiling data exported to {filename}.json")
    print(f"Memory profile exported to {filename}.html")

print("\nGENERATED TEXTS:")
for i, (input_text, generated_text) in enumerate(zip(batch_texts, generated_texts)):
    print(f"Input {i+1}: {input_text}")
    print(f"Output {i+1}: {generated_text}")
    print("-" * 50)