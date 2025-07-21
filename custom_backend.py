import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import List
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--compile", help = "compile mode", action='store_true')
args = parser.parse_args()
# include microbenchmarking

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

class GraphBreakModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        y = torch.sin(x)
        return torch.cos(y)

def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32),
        torch.randint(1000, (b,)),
    )

def test_module():
    model = GraphBreakModule()
    model.eval()
    data = generate_data(1)[0]
    with torch.no_grad():
        output = model(data)
    output_np = output.numpy()

def test_compiled_module():
    model = GraphBreakModule()
    model.eval()
    model.compile(backend=my_compiler)
    data = generate_data(1)[0]
    with torch.no_grad():
        compiled_output = model(data)
    compiled_np = compiled_output.numpy()

if __name__ == "__main__":
    if args.compile:
        test_compiled_module()
    test_module()
