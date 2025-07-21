### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manas-j/IndosysPytorchTuts.git
   cd IndosysPytorchTuts
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   <!-- ```bash
   pip install -r requirements.txt
   ``` -->
   Alternatively, if you prefer using the project configuration:
   ```bash
   pip install -e .
   ```

## How to Run

### 1. view FX graph using a custom backend(`custom_backend.py`)

Run with eager execution:
```bash
python custom_backend.py
```

Run with custom backend:
```bash
python custom_backend.py -c
```

### 2. Pointwise Fusion of ops (`pointwise_fusion.py`)

Run basic pointwise operations:
```bash
python pointwise_fusion.py
```

Run with compilation to see fusion effects:
```bash
python pointwise_fusion.py -c
```

### 3. Basic Inference script (`inference.py`)

Run model inference in eager mode:
```bash
python inference.py
```

Run with compilation enabled:
```bash
python inference.py -c
```

### 4. Profiling and Tracing (`hf_granite8b_w_trace.py`)

Run profiling in eager mode:
```bash
python hf_granite8b_w_trace.py
```

Run profiling with compilation:
```bash
python hf_granite8b_w_trace.py -c
```

**Viewing traces:**
- Open `.json` files in Perfetto `https://ui.perfetto.dev/` tool
- Open `.html` files directly in your browser

### 5. Benchmarking (`hf_granite8_latency.py`)

Run latency tests in eager mode:
```bash
python hf_granite8_latency.py
```

Run with compilation:
```bash
python hf_granite8_latency.py -c
```

# Todo:
    > show different type of gbs:
        > dynamic shapes, using guards to avoid recompilations (use hooks)
    > tlparsecd pt  
    > add mobilenet colab: https://colab.research.google.com/drive/1xmb4hXynYpvyk59chT8MsdaY3mqJHwvb?usp=sharing