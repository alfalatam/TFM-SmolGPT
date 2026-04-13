# SMOLGPT for computer vision 🦾

This repository is a rewrite of [smolGPT](https://github.com/Om-Alve/smolGPT), a minimal PyTorch implementation for training your own small LLM from scratch. Designed for educational purposes and simplicity, featuring efficient training, flash attention, and modern sampling techniques, this one in particular has been modified to train for computer vision purposes. This implementation is inspired by modern LLM training practices and adapted for educational purposes.

## Features ✨

- **Minimal Codebase**: Pure PyTorch implementation with no abstraction overhead
- **Modern Architecture**: GPT model with:
  - Flash Attention (when available)
  - RMSNorm and SwiGLU
  - Efficient top-k/p/min-p sampling
  - Rotary embeddings - RoPE (Optional)
- **Training Features**:
  - Mixed precision (bfloat16/float16)
  - Gradient accumulation
  - Learning rate decay with warmup
  - Weight decay & gradient clipping
- **Dataset Support**: Built-in TinyStories dataset processing
- **Custom Tokenizer**: SentencePiece tokenizer training integration

## Installation 🛠️

It is highly recommended to install an virtual enviroment.

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / Mac

python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

### Data Pipeline

The training pipeline follows these steps:

1. Raw text dataset (`.txt`)
2. Tokenizer training (SentencePiece)
3. Tokenization into `.bin` format 
4. Model training
5. Inference


**Requirements**:

- Python 3.10+
- NumPy 2.0.2
- Requests 2.32.3
- SentencePiece 0.2.0
- TensorBoard 2.18.0
- PyTorch 2.6.0
- tqdm 4.67.1

- Modern GPU with CUDA(recommended) or you can use Google Colab free resources

## GPU Support

Although the project can run on CPU, GPU acceleration is recommended for better training

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118


## GPU Support (Optional)

Although project can run on CPU, the GPU acceleration is highly recommended.

To enable GPU support, install PyTorch with CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## CLI Usage ⚙️

To see all available help commands:

```bash
python -m smolgpt.tools --help
```
## Quick Start 🚀

### Full Training Cycle

1. **Prepare Dataset**
```bash
python -m smolgpt.tools train_vocab_txt --vocab-size 4096 --txt-dir data/libros
```

2. **Start Training**
```bash
python -m smolgpt.train
```

*Training and validation loss are logged in `out/logs/`. To visualize using TensorBoard, run:*
```bash
tensorboard --logdir=out/logs
```

3. **Generate Text**

```bash
To start using the provided model it is necessary to uncompress it in `out` folder
```

```bash
python -m smolgpt.tools ask
```




## Pre-trained Model Details 🔍

The provided checkpoint was trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Architecture:
- 4096-token vocabulary
- 8 heads
- 8-layer transformer
- 512 embedding dimension
- Trained on `~1.5 Billion Tokens` for around `18.5` hours

Validation Loss - `1.0491`
<!-- [Loss Curve](assets/loss.png) -->

## Sample Outputs 📝

### Example 1
```text
Prompt: In computer vision, what is the RGB (Red, Green, Blue) color space?

Output:
In computer vision, the system of representing colors using three primary components: red, green, and blue light intensities. Each component corresponds to a specific wavelength within the visible spectrum, with each pixel's intensity representing the strength of that particular wavelength. This system allows computers to represent and manipulate images by combining these individual wavelengths into different shades and hues. A common application of this model is in digital cameras, where RGB sensors capture the varying amounts of red, green, and blue light from the scene to create a visual representation on a screen or storage device.


```

```
Prompt: what is the main idea behind Otsu’s thresholding method?

Output:
Answer: Otsu's thresholding method is an image segmentation technique that aims to automatically determine the optimal threshold value for dividing the image into two distinct regions based on their pixel intensities. The core principle of this method lies in finding the threshold that minimizes the variance within each region after applying it. This means maximizing the difference between the intensity values of foreground and background pixels while minimizing the overlap between them. For instance, consider medical imaging where Otsu's threshold can be used to segment tumors from healthy tissue by identifying areas with higher contrast..

```

## Configuration ⚙️

Key parameters (modify in `config.py`):

**Model Architecture**:
```python
GPTConfig(
    block_size=512,    # Context length
    n_layer=8,         # Number of transformer layers
    n_head=8,          # Number of attention heads
    n_embed=512,       # Embedding dimension
    dropout=0.0,       # Dropout rate
    bias=False,        # Use bias in layers
    use_rotary=False,  # Toggle rotary embeddings
)
```

**Training**:
```python
TrainingConfig(
    learning_rate: float = 6e-4
    max_iters: int = 10000

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 400   
    lr_decay_iters: int = 10000
    min_lr: float = 6e-5

    eval_interval: int = 500  
    log_interval: int = 50
    eval_iters: int = 30

    gradient_accumulation_steps: int = 2
    batch_size: int = 64

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "float16"
    compile: bool = False
)
```

## File Structure 📁

```smolgpt-computer-vision/
|
├── pyproject.toml - Project configuration & dependencies
├── assets/ - Images and resources for documentation
├── data/ - Training data (txt / bin files)
├── notebooks/ - Colab notebooks for Google Colab
├── out/ - Model checkpoints and logs
├── src/
│ └── smolgpt/ - Core package
│   ├── config.py - Model & training configuration
│   ├── model.py - GPT architecture implementation
│   ├── tokenizer.py - Tokenizer (SentencePiece wrapper)
│   ├── dataset.py - Data loading & batching
│   ├── train.py - Training loop
│   ├── sample.py - Text generation / inference
│   ├── evaluation.py - Test type (In progress..)
│   └── tools/ - CLI utilities (tokenizer,ask,check vocabulary...)
└── tests/ - Json test
```


### Training RIG SPECS (Free resources used via Google Colab)  

The model has been trained using free computational resources provided by Google Colab, demonstrating that it is possible to develop and fine-tune compact language models without requiring expensive infrastructure.

Additionally, the training process can be reproduced locally using consumer-grade hardware. In particular, a GPU with 6 GB of VRAM such as an **NVIDIA RTX 3060** is enough to run the training with the current configuration.

This highlights the focus of the project on efficiency and accessibility, enabling experimentation with GPT models under limited computational resources.


- **GPU**: NVIDIA GPU T4  
- **vCPUs**: 16  
- **RAM**: 12 GB  
- **VRAM**: 15 GB  
---


## Limitations ⚠️

Due to the small size of the model and the dataset, the system may:
- Generate repetitive outputs
- Lack deep reasoning capabilities
- Be more sensitive to the prompt phrasing

These limitations are inherent to small-scale autoregressive models.



## Acknowledgements 🙌

This project is based on an implementation inspired by SmolGPT.

Special thanks to [Om-Alve](https://github.com/Om-Alve) for their excellent work on building a GPT-style model from scratch, which served as a foundation and reference for this project.

This repository extends and adapts that work towards a computer vision educational assistant, including custom dataset generation, fine-tuning strategies, and evaluation.

---
