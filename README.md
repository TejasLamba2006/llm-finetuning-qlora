# Fine-Tuning LLMs with QLoRA - Complete Guide

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Tutorial-red?style=for-the-badge&logo=youtube)](https://youtu.be/EYtGKh7kNMQ)
[![Open In Colab](https://img.shields.io/badge/Colab-Open%20Notebook-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/drive/1_XzzJCWiuKPzKYIHNqRwoR5H5k_KAUQR?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Complete step-by-step guide to fine-tune Large Language Models using QLoRA technique. Tutorial in Hinglish (English + Hindi)!**

Created by [TejasLamba2006](https://github.com/TejasLamba2006) | Complete YouTube Tutorial Available!

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [What You'll Learn](#-what-youll-learn)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Detailed Explanation](#-detailed-explanation)
- [Results](#-results)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Overview

This repository contains a complete implementation of **Large Language Model Fine-Tuning** using:

- **Model**: Llama 3.2 3B Instruct
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Dataset**: FineTome-100k (High-quality instruction dataset)
- **Library**: Unsloth (2x faster training, 60% less memory)
- **Hardware**: Free T4 GPU on Google Colab (or local GPU)

### Full Video Tutorial

Watch the complete tutorial in Hinglish on YouTube: [**INSERT YOUR VIDEO LINK**]

---

## Features

- ✅ **Memory Efficient**: Uses 4-bit quantization (QLoRA)
- ✅ **Fast Training**: Unsloth library provides 2x speedup
- ✅ **Free GPU Compatible**: Runs on Google Colab free tier
- ✅ **Production Ready**: Complete code with best practices
- ✅ **Well Documented**: Every line explained in detail
- ✅ **Bilingual Tutorial**: Hinglish explanations in video script
- ✅ **Fully Reproducible**: Step-by-step instructions

---

## What You'll Learn

### Concepts Covered

1. **Fine-Tuning Fundamentals**
   - What is Fine-Tuning?
   - When to use Fine-Tuning?
   - Supervised Fine-Tuning (SFT)

2. **Techniques Comparison**
   - Full Fine-Tuning
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)

3. **Technical Deep Dives**
   - Attention Mechanisms (Q, K, V Projections)
   - 4-bit Quantization
   - PEFT (Parameter Efficient Fine-Tuning)
   - Chat Templates & Tokenization

4. **Practical Implementation**
   - Model Loading & Configuration
   - Dataset Processing
   - Training Loop
   - Hyperparameter Tuning
   - Model Saving & Inference

5. **Production Best Practices**
   - Memory Optimization
   - Gradient Accumulation
   - Mixed Precision Training
   - Checkpoint Management

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 16GB+ VRAM (or use Google Colab free T4)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB free space

### Software

- **Python**: 3.8+
- **CUDA**: 11.8+ (for local training)

### Libraries

```bash
unsloth
transformers
trl
torch
datasets
```

---

## Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1. **Open the notebook in Colab:**

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_XzzJCWiuKPzKYIHNqRwoR5H5k_KAUQR?usp=sharing)

2. **Enable GPU:**

   ```
   Runtime → Change runtime type → T4 GPU
   ```

3. **Run all cells!** 

### Option 2: Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TejasLamba2006/llm-finetuning-qlora.git
   cd llm-finetuning-qlora
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install unsloth transformers trl torch datasets
   ```

4. **Run the notebook:**

   ```bash
   jupyter notebook Finetuning_Llama3_2_3B.ipynb
   ```

---

## Project Structure

```
llm-finetuning-qlora/
│
├── Finetuning_Llama3_2_3B.ipynb    # Main training notebook
├── visa2code_youtube_script.md     # Complete video script (Hinglish)
├── YOUTUBE_METADATA.md              # Video metadata and descriptions
├── README.md                        # This file
│
├── data/                            # Sample data (optional)
│   └── hawaii_wf_*.txt
│
├── outputs/                         # Training outputs (created during training)
│   └── checkpoint-*/
│
├── finetuned_model/                # Saved LoRA adapters (created after training)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
│
└── requirements.txt                # Python dependencies
```

---

## Detailed Explanation

### Step 1: Installation

```python
%pip install unsloth transformers trl
```

- **unsloth**: Optimized for Llama models, 2x faster training
- **transformers**: Hugging Face library for model architectures
- **trl**: Supervised Fine-Tuning trainer

### Step 2: Model Loading

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True  # 4-bit quantization for memory efficiency
)
```

**Key Parameters:**

- `max_seq_length`: Maximum tokens (2048 ≈ 1500 words)
- `load_in_4bit`: Enables QLoRA (87.5% memory savings!)

### Step 3: PEFT Configuration

```python
model = FastLanguageModel.get_peft_model(
    model, 
    r=16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
)
```

**Target Modules:**

- **Attention**: q_proj, k_proj, v_proj, o_proj
- **Feed-Forward**: gate_proj, up_proj, down_proj

**Result**: Only 0.3% parameters are trainable! 🎉

### Step 4: Dataset Processing

```python
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
```

**Dataset**: 100k high-quality instruction-following examples

### Step 5: Training

```python
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        ...
    )
)
trainer.train()
```

**Key Hyperparameters:**

- **Learning Rate**: 2e-4 (optimal for LoRA)
- **Batch Size**: 2 (effective: 2×4=8 with gradient accumulation)
- **Steps**: 60 for demo (use 1000+ for production)

### Step 6: Inference

```python
response = inference_model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
```

---

## 📊 Results

### Training Metrics

- **Initial Loss**: ~2.5
- **Final Loss**: ~0.5
- **Training Time**: ~3-5 minutes (60 steps on T4 GPU)
- **Memory Usage**: ~6-8 GB

### Model Size

- **Base Model**: ~6 GB
- **LoRA Adapters**: ~30 MB (200x smaller!)

### Sample Output

```
Prompt: "What are the key principles of investment?"

Response: "The key principles of investment include:
1. Diversification - spreading investments across different assets
2. Risk Management - understanding your risk tolerance
3. Long-term Perspective - investing is a marathon, not a sprint
..."
```

---

## Deployment

### Option 1: Save Locally

```python
model.save_pretrained("finetuned_model")
```

### Option 2: Upload to Hugging Face Hub

```python
model.push_to_hub("your-username/your-model-name")
```

### Option 3: Merge and Export

```python
model = model.merge_and_unload()
model.save_pretrained("merged_model")
```

### Inference in Production

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    load_in_4bit=True
)
```

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# Reduce batch size
per_device_train_batch_size=1

# Reduce sequence length
max_seq_length=1024

# Increase gradient accumulation
gradient_accumulation_steps=8
```

### Issue 2: Slow Training

**Solutions**:

- Enable `bf16=True` (if GPU supports it)
- Reduce `logging_steps=10`
- Use smaller dataset for testing

### Issue 3: Poor Quality Outputs

**Solutions**:

- Train longer (increase `max_steps`)
- Use better quality dataset
- Adjust `learning_rate` (try 1e-4 or 3e-4)
- Increase LoRA `r=32`

### Issue 4: Installation Errors

**Solutions**:

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other libraries
pip install unsloth transformers trl
```

---

## Resources

### Documentation

- **Unsloth**: <https://unsloth.ai/>
- **Transformers**: <https://huggingface.co/docs/transformers>
- **TRL**: <https://huggingface.co/docs/trl>

### Dataset

- **FineTome-100k**: <https://huggingface.co/datasets/mlabonne/FineTome-100k>

### Learning Resources

- **Attention Mechanisms**: <https://medium.com/@kalra.rakshit/introduction-to-transformers-and-attention-mechanisms-c29d252ea2c5>
- **Original Paper** (Attention is All You Need): <https://arxiv.org/abs/1706.03762>
- **LoRA Paper**: <https://arxiv.org/abs/2106.09685>
- **QLoRA Paper**: <https://arxiv.org/abs/2305.14314>

### Related Videos

- **Full Tutorial**: [YOUR YOUTUBE VIDEO]
- **Next: DPO Tutorial**: Coming Soon!

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Areas for Contribution

- [ ] Add support for other models (Mistral, Qwen, etc.)
- [ ] Implement DPO fine-tuning
- [ ] Add evaluation metrics
- [ ] Create dataset creation guide
- [ ] Add more language support in scripts
- [ ] Improve documentation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Model weights and datasets have their own licenses:

- Llama 3.2: [Meta's Llama License](https://llama.meta.com/llama-downloads/)
- FineTome-100k: Check Hugging Face dataset page

---

## Acknowledgments

- **Meta AI** for Llama 3.2 model
- **Hugging Face** for Transformers, TRL, and Datasets libraries
- **Unsloth** team for the amazing optimization library
- **mlabonne** for the FineTome-100k dataset
- **All contributors** and the open-source community

---

## 💬 Contact & Support

- **YouTube**: [visa2code](https://www.youtube.com/@visa2code)
- **GitHub**: [TejasLamba2006](https://github.com/TejasLamba2006)
- **LinkedIn**: [Your Profile](https://www.linkedin.com/in/tejaslamba/)
- **Discord**: [Join Community](https://discord.gg/msEkYDWpXM)

### Got Questions?

- Comment on the YouTube video
- Open an issue on GitHub
- Email: <your.email@example.com>

---

## Show Your Support

If this project helped you, please:

- ⭐ Star this repository
- 👍 Like the YouTube video
- 🔔 Subscribe to visa2code channel
- 🔗 Share with your network
- 💬 Leave a comment

---

## 📈 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/TejasLamba2006/llm-finetuning-qlora?style=social)
![GitHub forks](https://img.shields.io/github/forks/TejasLamba2006/llm-finetuning-qlora?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/TejasLamba2006/llm-finetuning-qlora?style=social)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{visa2code-llm-finetuning,
  author = {visa2code},
  title = {Fine-Tuning LLMs with QLoRA - Complete Guide},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/TejasLamba2006/llm-finetuning-qlora}
}
```

---

## 🗺️ Roadmap

- [x] Basic fine-tuning implementation
- [x] Complete documentation
- [x] Hinglish tutorial script
- [ ] DPO implementation
- [ ] Multi-GPU support
- [ ] Custom dataset creation guide
- [ ] Model evaluation metrics
- [ ] Deployment guide (vLLM, TGI)
- [ ] AWS/GCP deployment tutorial
- [ ] More model support (Mistral, Qwen, etc.)

---

<div align="center">

**Made with ❤️ by [visa2code](https://www.youtube.com/@visa2code)**

**Subscribe for more AI tutorials in Hinglish!**

[![YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/TejasLamba2006/llm-finetuning-qlora)

</div>

---

## 📝 Changelog

### Version 1.0.0 (2026-02-14)

- ✨ Initial release
- ✅ Complete fine-tuning implementation
- ✅ Hinglish video script
- ✅ Comprehensive documentation
- ✅ Google Colab notebook

---

**Happy Fine-Tuning! Keep Learning, Keep Coding! 🚀**
