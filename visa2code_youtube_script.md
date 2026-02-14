# Fine-Tuning LLMs - Complete Guide | visa2code

## YouTube Script - English + Hindi (Hinglish)

---

## INTRODUCTION (0:00 - 1:30)

**Namaste dosto! Aur hello everyone!**

Welcome back to **visa2code**! Aaj ke video mein hum seekhenge **Large Language Models ko fine-tune karna** - completely from scratch!

Agar aapko kabhi ye doubt aaya hai ki:

- Fine-tuning kya hoti hai aur kyun zaroori hai?
- LoRA, QLoRA ye kya cheezein hain?
- Apni khud ki custom AI model kaise banayein?
- Production-grade fine-tuning kaise karein?

Toh ye video aapke liye **perfect** hai! Main aapko **har ek line of code** explain karunga, **har parameter**, **har library** - **complete detail mein**!

Toh chaliye shuru karte hain aur **code along** zaroor karein mere saath!

---

## PART 1: UNDERSTANDING FINE-TUNING (1:30 - 4:00)

### What is Supervised Fine-Tuning?

Dosto, pehle samajhte hain **fine-tuning ka concept**.

**Fine-tuning ek method hai** jisse hum pre-trained LLMs ko improve aur customize kar sakte hain.

**Main goal kya hai?**

- Naye knowledge add karna NAHI hai
- Instead, hum ek base model ko transform karte hain jo sirf text predict karta tha
- Usko ek **assistant** mein convert karte hain jo:
  - Instructions follow kar sake
  - Questions answer kar sake
  - Specific tasks perform kar sake

**Real-world example:**
Suppose aapki company Excel documents ke saath kaam karti hai:

- Excel generation
- Excel explanation  
- Data analysis from spreadsheets

Toh aapko ek **specialized model** chahiye jo specifically **Excel tasks mein expert** ho. Generic GPT use karna expensive aur less accurate hoga. **Isliye fine-tuning!**

**ImportantPoint:** Fine-tuning tab best kaam karti hai jab:

- Base model mein already **related knowledge** present ho
- Aap completely naya domain nahi sikha rahe
- Example: Coding tasks ke liye coding model use karein (like Llama for code)
- Medical tasks ke liye medical knowledge wala model use karein

---

## PART 2: FINE-TUNING TECHNIQUES (4:00 - 7:00)

### Technique 1: Full Fine-Tuning

**Ye kya hai?**

- Poore model ke **saare parameters** ko retrain karna
- Sabse accurate results mil sakte hain
- **But problem kya hai?**
  - Bahut zyada GPU memory chahiye
  - Bahut zyada time lagta hai
  - Bahut expensive hai
  - **Catastrophic forgetting** ho sakti hai - model apna original knowledge bhool sakta hai

**Verdict:** Companies ke liye bhi impractical hai, toh hum **NAHI use karenge**!

---

### Technique 2: LoRA (Low-Rank Adaptation)

**Ye sabse popular technique hai!**

**LoRA kaise kaam karta hai?**

1. Original model weights ko **freeze** kar deta hai (unhe change nahi karta)
2. Each targeted layer mein **small adapter matrices** add karta hai
3. Ye adapters **low-rank matrices** hote hain - matlab bahut chhote!

**Benefits:**

- Sirf **less than 1%** parameters train hote hain
- Memory usage **drastically kam** ho jaati hai
- Training time bhi bahut **kam** lagta hai
- **Non-destructive** hai - original knowledge intact rahti hai
- Adapters ko switch ya combine kar sakte hain

**Example:**
Agar 7 billion parameter model hai:

- Full fine-tuning: 7 billion parameters train karenge
- LoRA: Sirf 7-10 million parameters train karenge (1% se bhi kam!)

---

### Technique 3: QLoRA (Quantized LoRA)

**Ye aur bhi efficient hai!**

**QLoRA = LoRA + Quantization**

**Kya hota hai?**

- Model weights ko **4-bit** mein store karta hai
- Normal precision: 32-bit ya 16-bit
- QLoRA: **4-bit** quantization
- Ye **33% extra memory reduction** deta hai compared to LoRA

**Real benefit:**

- GPU memory limited hai? No problem!
- Google Colab mein free GPU? No problem!
- Local laptop? No problem!
- **QLoRA use karo aur badi models bhi train karo!**

**Trade-off:**

- Memory efficiency: ✅ Excellent
- Speed: ✅ Fast
- Accuracy: ✅ Almost same as full fine-tuning
- Isliye **aaj hum QLoRA use karenge!**

---

## PART 3: LIBRARY INTRODUCTION (7:00 - 10:00)

### Why Unsloth Library?

Dosto, aaj hum **Unsloth library** use karenge. **Kyun?**

**Performance Benefits:**

- **2x faster** training compared to standard methods
- **60% less memory** usage
- **Llama models ke saath best compatibility**
- Production-ready aur battle-tested

**Alternative libraries:**

- Hugging Face PEFT
- Axolotl
- But Unsloth **sabse fast aur efficient** hai for Llama models!

**Aaj ka goal:**

- **Llama 3.2 3B Instruct model** ko fine-tune karenge
- **FineTome-100k dataset** use karenge
- **QLoRA technique** apply karenge
- Sirf **60 training steps** - demo ke liye
- Aap production mein **zyada steps** use kar sakte ho

---

## PART 4: ENVIRONMENT SETUP (10:00 - 12:00)

### GPU Setup (Show on screen)

**Important:** Ye code **GPU pe hi chalega**!

**Google Colab mein:**

```
Runtime → Change runtime type → T4 GPU (free)
```

**Local machine:**

- NVIDIA GPU with CUDA support
- At least 16GB VRAM for 3B model
- Smaller models: 8GB VRAM enough

**Kaggle:**

- Free GPU available
- Similar process as Colab

---

## PART 5: CODE WALKTHROUGH - INSTALLATION (12:00 - 14:00)

Chaliye ab **actual coding** shuru karte hain!

### Cell 1: Library Installation

```python
%pip install unsloth transformers trl
```

**Line-by-line explanation:**

**`%pip install`**

- Ye Jupyter notebook magic command hai
- Terminal mein `pip install` ke equivalent
- % symbol notebook-specific command indicate karta hai

**`unsloth`**

- Ye **main library** hai jo hum use kar rahe hain
- Optimized for Llama model fine-tuning
- Provides FastLanguageModel class
- Memory-efficient implementations
- **Must-have** for efficient fine-tuning

**`transformers`**

- **Hugging Face ka flagship library**
- Provides:
  - Model architectures (BERT, GPT, Llama, etc.)
  - Tokenizers - text ko numbers mein convert karte hain
  - Training utilities
  - Pre-trained model weights
- Industry standard library - **har jagah use hota hai**

**`trl`** (Transformer Reinforcement Learning)

- **Hugging Face ka library** for fine-tuning
- Provides **SFTTrainer** (Supervised Fine-Tuning Trainer)
- Handles:
  - Training loop
  - Gradient computation
  - Loss calculation
  - Optimization
- **Bahut powerful** aur easy to use

**After installation:**

- Restart runtime/kernel (Colab mein automatically suggest hota hai)
- Ye ensure karta hai packages properly load hon

---

## PART 6: CODE WALKTHROUGH - IMPORTS (14:00 - 18:00)

### Cell 2: Importing Libraries

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
```

**Har ek import ko detail mein samajhte hain:**

---

**`import torch`**

**Ye kya hai?**

- **PyTorch** - deep learning framework
- Facebook/Meta ne develop kiya
- Industry standard for AI/ML

**Ye kyun zaroori hai?**

- Sab kuch PyTorch tensors pe work karta hai
- GPU operations handle karta hai
- Neural network computations
- Backpropagation aur gradient descent

**Is video mein kahan use hoga?**

- Check karne ke liye bf16 support hai ya nahi
- CUDA availability check
- Tensor operations during training

---

**`from unsloth import FastLanguageModel`**

**FastLanguageModel kya hai?**

- Unsloth ka **optimized model loader**
- Standard AutoModel se **better performance**
- Features:
  - Faster inference
  - Less memory usage
  - 4-bit quantization support built-in
  - Flash Attention support

**Comparison:**

```
Standard loading: 30 seconds, 7GB memory
FastLanguageModel: 10 seconds, 4GB memory
```

**Methods jo hum use karenge:**

- `from_pretrained()` - model load karne ke liye
- `get_peft_model()` - LoRA adapters add karne ke liye

---

**`from datasets import load_dataset`**

**Datasets library kya hai?**

- Hugging Face ka data loading library
- **35,000+ datasets** available on Hugging Face Hub
- Features:
  - Automatic caching
  - Memory mapping - badi datasets efficiently handle karta hai
  - Streaming support
  - Fast processing with Arrow backend

**Load_dataset function:**

- Directly Hugging Face Hub se dataset download karta hai
- Automatically format detect karta hai
- Kaggle, CSV, JSON, Parquet - sab supported

**Example:**

```python
# Internet se download
dataset = load_dataset("username/dataset-name")

# Local file se load
dataset = load_dataset("csv", data_files="train.csv")
```

---

**`from trl import SFTTrainer`**

**SFTTrainer kya hai?**

- **Supervised Fine-Tuning Trainer**
- Specifically designed for instruction-following tasks
- Standard Trainer ka **specialized version**

**Kya handle karta hai?**

1. **Training loop** - automatically epochs chalata hai
2. **Loss computation** - next token prediction loss
3. **Gradient accumulation** - multiple batches combine karta hai
4. **Mixed precision training** - fp16/bf16 support
5. **Logging** - training metrics track karta hai
6. **Checkpointing** - model save karta hai during training

**Behind the scenes:**

- Dataset ko tokenize karta hai
- Batches banata hai
- Forward pass
- Loss calculation
- Backward pass
- Optimizer step
- Ye sab **automatically**!

---

**`from transformers import TrainingArguments`**

**TrainingArguments kya hai?**

- Training ke **saare settings/configurations** yahan define karte hain
- One place mein **sab hyperparameters**

**Kya kya configure kar sakte hain?**

- Batch size
- Learning rate
- Number of epochs/steps
- Gradient accumulation
- Mixed precision (fp16/bf16)
- Logging frequency
- Output directory
- Learning rate scheduler
- Warmup steps
- Weight decay
- Aur **100+ options!**

**Example:**

```python
args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    num_train_epochs=3
)
```

---

**`from unsloth.chat_templates import get_chat_template, standardize_sharegpt`**

**Ye sabse interesting part hai! Dhyan se samjho:**

**Chat templates kya hain?**

Jab hum AI se baat karte hain, toh ye format hota hai:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
]
```

**But model ye dictionary nahi samajh sakta!** Model ko **plain text** chahiye.

**Chat template ka kaam:**
Dictionary format ko **model-specific string** mein convert karna.

**Example - Llama 3.1 template:**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is Python?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Python is a programming language...<|eot_id|>
```

Har model ka **alag template** hota hai:

- Llama: special tokens like `<|begin_of_text|>`
- GPT: different format
- Claude: different format

---

**`get_chat_template` function:**

- Tokenizer mein **correct template** set karta hai
- Model-specific formatting ensure karta hai

**`standardize_sharegpt` function:**

- **ShareGPT format** ko standard format mein convert karta hai
- ShareGPT kya hai? Popular dataset format:

```json
{
  "conversations": [
    {"from": "human", "value": "Question here"},
    {"from": "gpt", "value": "Answer here"}
  ]
}
```

- Isko convert karta hai standard "role" format mein:

```json
{
  "conversations": [
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ]
}
```

**Kyun zaroori hai?**

- Bina proper formatting ke model **confused** ho jayega
- Training mein **errors** aayengi
- Model properly nahi seekhega instructions follow karna

---

## PART 7: MODEL LOADING (18:00 - 23:00)

### Cell 3: Loading Model and Tokenizer

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
```

**Is code ko line by line samajhte hain:**

---

**`model, tokenizer = ...`**

- **Tuple unpacking** ho rahi hai
- Function **do cheezein return** kar raha hai:
  1. `model` - actual language model
  2. `tokenizer` - text processor

---

**`FastLanguageModel.from_pretrained()`**

**Ye method kya karta hai?**

- Hugging Face Hub se model **download** karta hai
- Memory mein **load** karta hai
- Optimizations **apply** karta hai

**Behind the scenes:**

1. Model weights download (first time only, phir cache)
2. Model architecture load
3. Tokenizer files load
4. Quantization apply (if specified)
5. GPU pe move karna

---

**`model_name = "unsloth/Llama-3.2-3B-Instruct"`**

**Is naam ko breakdown karte hain:**

**`unsloth/`** - Hugging Face username/organization

- Unsloth team ne optimized version upload kiya hai
- Original bhi download kar sakte ho: `meta-llama/Llama-3.2-3B-Instruct`
- But Unsloth version **faster loading** aur **better optimized**

**`Llama-3.2`** - Model version

- Meta/Facebook ka latest model series
- Previous: Llama 3.1, Llama 3, Llama 2
- 3.2 has **better instruction following**

**`3B`** - Model size

- **3 Billion parameters**
- Other sizes available:
  - 1B - smallest, fastest
  - 8B - better quality
  - 70B - production quality (but needs more GPU)
  - 405B - best quality (cloud only)

**`Instruct`** - Model type

- Pre-trained on **instruction datasets**
- Knows how to follow commands
- Better for chat/assistant tasks
- Alternative: base models (without Instruct) - less aligned

**Kyun 3B choose kiya?**

- Balance between **quality** aur **resource requirements**
- Google Colab free tier mein chalega
- Decent performance for learning
- Production mein 8B ya 70B use karo

---

**`max_seq_length=2048`**

**Sequence length kya hai?**

- Maximum number of **tokens** model ek baar mein process kar sakta hai
- Token ≈ 0.75 words (English mein)
- 2048 tokens ≈ 1500-1600 words

**Context window:**

```
User message: 500 tokens
Previous messages: 800 tokens  
System prompt: 100 tokens
Assistant response: 648 tokens
----------------------
Total: 2048 tokens (limit)
```

**Kya hota hai agar exceed ho jaye?**

- Older messages **truncate** ho jayenge
- Model purani context "bhool" jayega
- Isliye limit ke andar rehna zaroori hai

**Can we increase it?**

- Haan, but:
  - More memory usage
  - Slower training
  - Some models support 4096, 8192, even 128k
- **Trade-off:** Speed vs Context

**Best practice:**

- Training ke liye: 2048 enough
- Production inference: aapki need ke according adjust karo

---

**`load_in_4bit=True`**

**Ye sabse important parameter hai!**

**Quantization kya hai?**

- Model weights ko **compress** karna
- Less precision use karna numbers store karne ke liye

**Precision levels:**

**32-bit (Full Precision):**

- Number: 3.14159265358979323846
- Memory: 4 bytes per parameter
- Accuracy: Maximum
- Use: Research, final evaluation

**16-bit (Half Precision - fp16/bf16):**

- Number: 3.141593
- Memory: 2 bytes per parameter (50% savings)
- Accuracy: Almost same
- Use: Standard training

**4-bit (Quantization):**

- Number: Limited precision
- Memory: 0.5 bytes per parameter (87.5% savings!)
- Accuracy: Slightly lower but acceptable
- Use: **Resource-constrained training/inference**

**Real-world example:**

3B model memory usage:

- 32-bit: 3B params × 4 bytes = **12 GB**
- 16-bit: 3B params × 2 bytes = **6 GB**
- 4-bit: 3B params × 0.5 bytes = **1.5 GB** ✅

**Plus gradients aur activations:**

- 32-bit training: ~25-30 GB needed
- 4-bit with QLoRA: **~6-8 GB needed**

**Isliye Colab free GPU (15GB) mein chalega! 🎉**

**Accuracy impact:**

- 99% tasks mein **negligible difference**
- Production models bhi use karti hain
- Trade-off totally **worth it**

---

## PART 8: PEFT CONFIGURATION (23:00 - 30:00)

### Cell 4: Configuring LoRA/PEFT

```python
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
)
```

**Ye sabse technical part hai! Dhyan se suno:**

---

**`FastLanguageModel.get_peft_model()`**

**PEFT kya hai?**

- **Parameter-Efficient Fine-Tuning**
- Saare parameters train karne ki bajaye
- Sirf **chhote adapters** train karte hain
- Big savings: memory, time, cost

**Ye function kya karta hai?**

1. Original model weights ko **freeze** karta hai
2. Selected layers mein **LoRA adapters** inject karta hai
3. Sirf adapters trainable hote hain
4. Original model **unchanged** rehta hai

---

**`model` parameter**

- Pehle load kiya hua model pass kar rahe hain
- Isi model mein adapters add honge

---

**`r=16` - RANK parameter**

**Ye sabse important hyperparameter hai!**

**Rank kya hai? (Simple explanation):**

LoRA adapters do choti matrices use karte hain: **A** aur **B**

Original weight: W (bahut bada matrix)
LoRA addition: Δ W = A × B

**Rank (r)** decides kitne "features" adapters seekh sakte hain.

**Size comparison:**

Suppose original weight matrix: 4096 × 4096 = ~16 million values

**With LoRA r=16:**

- Matrix A: 4096 × 16 = 65,536 values
- Matrix B: 16 × 4096 = 65,536 values
- Total: 131,072 values (99% reduction!)

**Different rank values:**

**r=8** (Low rank):

- Least parameters
- Fastest training
- Least memory
- But: **Limited learning capacity**
- Use: Simple tasks, very limited GPU

**r=16** (Medium rank) ✅ **Most common**:

- Balanced parameters
- Good learning capacity
- Reasonable memory
- Best **accuracy vs efficiency** trade-off
- Use: **Most use cases** - ye default choice hai

**r=32** (High rank):

- More parameters
- Better learning capacity
- More memory usage
- Slightly better accuracy
- Use: Complex tasks, sufficient GPU

**r=64+** (Very high):

- Approaching full fine-tuning benefits
- But bhi zyada memory usage
- Rarely needed

**Best practice:** Start with 16, adjust if needed

---

**`target_modules` - Adapter injection points**

**Ye kaunse layers hain? Attention mechanism se related!**

Pehle samajhte hain **Transformer architecture**:

**Attention mechanism:**

- "Attention is All You Need" paper (2017)
- Foundation of all modern LLMs
- Samajhta hai words ke beech relations

**Transformer ka ek layer:**

```
Input
  ↓
Multi-Head Attention  ← Yahan adapters lagenge!
  ↓
Feed-Forward Network  ← Yahan bhi adapters lagenge!
  ↓
Output
```

---

**Attention modules:**

**1. `q_proj` - Query Projection**

**Kya hai?**

- "Q" in attention formula: Q, K, V
- Query matrix banata hai
- **"What am I looking for?"** - ye define karta hai

**Example:**
Sentence: "The cat sat on the mat"
Word "sat" ka query: "I need context about WHO sat and WHERE"

**Linear transformation:**

```
Input embedding → Query projection → Query vector
```

---

**2. `k_proj` - Key Projection**

**Kya hai?**

- "K" in attention formula
- Key matrix banata hai  
- **"What information do I have?"** - ye define karta hai

**Example:**
Word "cat" ka key: "I am an animal, I am the subject"
Word "mat" ka key: "I am an object, I am a location"

---

**3. `v_proj` - Value Projection**

**Kya hai?**

- "V" in attention formula
- Value matrix banata hai
- **"What actual information should I pass?"** - ye define karta hai

**Example:**
Word "cat" ka value: [embeddings representing cat features]

---

**Attention formula:**

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**Process:**

1. Query aur Keys ka **similarity** calculate karo
2. Softmax se **attention weights** nikalo
3. Values ko **weighted sum** karo

**Result:** Har word ko context-aware representation milta hai

---

**4. `o_proj` - Output Projection**

**Kya hai?**

- Attention ke baad **final transformation**
- Multi-head attention outputs ko combine karta hai
- Next layer ke liye prepare karta hai

---

**Feed-Forward Network modules:**

Har attention layer ke baad ek **Feed-Forward Network** hai:

```
FFN(x) = activation(x × W_gate) ⊙ (x × W_up) × W_down
```

**5. `gate_proj` - Gate Projection**

**Kya hai?**

- **Gating mechanism** for information flow
- Decide karta hai kaunsi information pass karni hai
- SwiGLU activation use karta hai (Llama mein)

**Intuition:**
"Is information ko kitna importance doon?"

---

**6. `up_proj` - Up Projection**

**Kya hai?**

- Dimension **expand** karta hai
- Usually 4x larger (example: 4096 → 16384)
- More capacity for transformations

**Why expand?**

- Richer representations
- Non-linear transformations
- Better feature learning

---

**7. `down_proj` - Down Projection**

**Kya hai?**

- Dimension wapas **compress** karta hai
- Back to original size (16384 → 4096)
- Output for next layer

---

**Kyun in modules ko target kiya?**

**Research findings:**

- Ye modules sabse zyada **impact** create karte hain
- Inko adapt karke model behavior significantly change hoti hai
- Other modules (LayerNorm, embeddings) ko freeze rakhna OK hai

**Comparison:**

**Only attention modules (q, k, v, o):**

- Faster training
- Less memory
- Good for: conversational patterns, instruction following

**Attention + FFN modules (all 7) ✅ - Ye hum use kar rahe hain:**

- Slower but still efficient
- More memory but manageable
- Best for: **Complex tasks, knowledge adaptation**

**All modules:**

- Slowest, approaching full fine-tuning
- Most memory
- Usually overkill

---

**Visual Summary:**

```
Transformer Layer:
┌─────────────────────┐
│   Input Tokens      │
└─────────┬───────────┘
          │
    ┌─────▼─────┐
    │ Attention │
    │ ┌───────┐ │
    │ │q_proj │ │ ← LoRA adapter
    │ │k_proj │ │ ← LoRA adapter  
    │ │v_proj │ │ ← LoRA adapter
    │ │o_proj │ │ ← LoRA adapter
    │ └───────┘ │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │    FFN    │
    │ ┌───────┐ │
    │ │ gate  │ │ ← LoRA adapter
    │ │  up   │ │ ← LoRA adapter
    │ │ down  │ │ ← LoRA adapter
    │ └───────┘ │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Output   │
    └───────────┘
```

**Memory calculation:**

3B model with LoRA:

- Frozen parameters: 3B (read-only)
- LoRA adapters (r=16, 7 modules): ~8-10M trainable
- **Ratio: 0.3% trainable parameters!**
- **But results: close to full fine-tuning!**

**Production tip:**
Experiment karo different target_modules ke saath:

```python
# Minimal (for limited GPU)
target_modules = ["q_proj", "v_proj"]

# Standard (balanced) ✅
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# Comprehensive (if GPU allows)  
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "embed_tokens", "lm_head"]
```

---

## PART 9: TOKENIZER CONFIGURATION (30:00 - 32:00)

### Cell 5: Setting Chat Template

```python
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

**Explanation:**

**`get_chat_template()` function:**

- Pehle load kiya hua tokenizer modify kar raha hai
- Specific **chat template** set kar raha hai
- Returns updated tokenizer

**`chat_template="llama-3.1"`:**

- Llama 3.1 ka **official template** use karega
- Special tokens properly set honge:
  - `<|begin_of_text|>`
  - `<|start_header_id|>`
  - `<|end_header_id|>`
  - `<|eot_id|>` (end of turn)
  - `<|end_of_text|>`

**Kyun zaroori hai?**

- Training aur inference mein **consistency** chahiye
- Model exactly wahi format seekhega jo wo expect karta hai
- Galat template = confusion = poor results

**Other available templates:**

- `"llama-3"` - older version
- `"chatml"` - for Qwen models
- `"alpaca"` - for Alpaca-style datasets
- `"vicuna"` - for Vicuna-style datasets

**Best practice:** Always use model-specific template!

---

## PART 10: DATASET LOADING (32:00 - 37:00)

### Cell 6 & 7: Loading and Standardizing Dataset

```python
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
```

**Breakdown:**

**`load_dataset()`:**

- Hugging Face Hub se dataset fetch karega
- Automatic caching - second time fast load hoga
- Memory-efficient loading with Apache Arrow

**`"mlabonne/FineTome-100k"`:**

**FineTome kya hai?**

- Curated instruction dataset
- **100,000 high-quality examples**
- Derived from **FineWeb-Edu** (Hugging Face ka massive dataset)
- Filtered for:
  - Quality
  - Diversity  
  - Educational value

**Dataset structure:**

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Explain what recursion is in programming"
    },
    {
      "from": "gpt", 
      "value": "Recursion is when a function calls itself..."
    }
  ]
}
```

**`split="train"`:**

- Sirf training split load kar rahe hain
- Kuch datasets mein multiple splits hote hain:
  - "train" - for training
  - "validation" - for validation during training
  - "test" - for final evaluation
- Ye dataset mein sirf train hai

---

```python
dataset = standardize_sharegpt(dataset)
```

**Ye step kya kar raha hai?**

**Before standardization:**

```json
{
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

**After standardization:**

```json
{
  "conversations": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

**Changes:**

- `"from": "human"` → `"role": "user"`
- `"from": "gpt"` → `"role": "assistant"`  
- `"value"` → `"content"`

**Kyun?**

- Standard OpenAI-style format
- Compatible with Llama chat template
- Industry standard format

---

### Cell 8 & 9: Inspecting Dataset

```python
dataset
```

**Output dikhega:**

```
Dataset({
    features: ['conversations'],
    num_rows: 100000
})
```

**Samajhne ke liye:**

- 100,000 training examples
- Each example has "conversations" field

```python
dataset[0]
```

**First example dekh sakte hain:**

```python
{
  'conversations': [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'The capital of France is Paris...'}
  ]
}
```

**Best practice:** Always inspect your data before training!

---

## PART 11: DATASET TOKENIZATION (37:00 - 42:00)

### Cell 10: Converting to Chat Format

```python
dataset = dataset.map(
    lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
    },
    batched=True
)
```

**Ye sabse complex transformation hai! Carefully samajhte hain:**

---

**`dataset.map()`**

**Kya hai?**

- Datasets library ka function
- Har example pe **transformation apply** karta hai
- New column add kar sakta hai
- Parallel processing - bahut **fast**!

**Parameters:**

- Function jo apply karna hai
- `batched=True` - multiple examples ek saath process karo

---

**`lambda examples: {...}`**

**Lambda function kya hai?**

- Anonymous function (bina naam ka)
- On-the-fly define karke use kar sakte hain

**`examples` kya hai?**

- Kyunki `batched=True`, ye **list of examples** hai
- Example:

```python
examples = {
    "conversations": [
        [{"role": "user", "content": "Q1"}, ...],  # Example 1
        [{"role": "user", "content": "Q2"}, ...],  # Example 2
        # ... more examples
    ]
}
```

---

**Return value: `{"text": [...]}`**

**New column create kar rahe hain:** `"text"`

**Kyun new column?**

- Original "conversations" structured format hai
- Training ke liye **plain text** chahiye
- "text" column mein tokenized strings hongi

---

**List comprehension:**

```python
[
    tokenizer.apply_chat_template(convo, tokenize=False)
    for convo in examples["conversations"]
]
```

**Breaking it down:**

**`for convo in examples["conversations"]`:**

- Har conversation pe loop kar rahe hain
- `convo` ek single conversation hai (list of messages)

**`tokenizer.apply_chat_template(convo, tokenize=False)`:**

**Ye sabse important function hai!**

**Kya karta hai?**

1. Conversation (list of dicts) leta hai
2. Model-specific format mein convert karta hai
3. Special tokens add karta hai
4. Single string return karta hai

**`tokenize=False` kyun?**

- Abhi sirf string chahiye, tokens nahi
- Actual tokenization trainer automatically karega
- Isliye plain text string return karega

---

**Example transformation:**

**Input (convo):**

```python
[
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."}
]
```

**Output (text string):**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful<|eot_id|><|start_header_id|>user<|end_header_id|>

What is AI?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

AI is...<|eot_id|>
```

**Ye string exactly wahi hai jo model training mein dekhega!**

---

**After this transformation:**

**Dataset format:**

```python
{
  'conversations': [...],  # Original (still there)
  'text': '<|begin_of_text|>...<|eot_id|>'  # New formatted string
}
```

---

### Cell 11 & 12: Verifying Transformation

```python
dataset
```

**Output:**

```
Dataset({
    features: ['conversations', 'text'],  # Now 2 columns!
    num_rows: 100000
})
```

```python
dataset[0]
```

**Ab kya dikhega:**

```python
{
  'conversations': [original conversation list],
  'text': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>...'
}
```

**Verification:**

- Check karo "text" field properly formatted hai
- Special tokens present hain
- Structure correct hai

**Agar errors hain:**

- Chat template galat ho sakta hai
- Tokenizer properly configured nahi hai
- Dataset format unexpected hai

---

## PART 12: TRAINING CONFIGURATION (42:00 - 52:00)

### Cell 13: Setting up Trainer

```python
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs"
    ),
)
```

**Har parameter ko deeply samajhte hain:**

---

**`model = model`**

- PEFT-enabled model pass kar rahe hain
- Wo model jisme LoRA adapters hain

**`train_dataset = dataset`**

- Training data
- 100k formatted examples

**`dataset_text_field = "text"`**

- Konse column se data lena hai
- Humne "text" column create kiya tha with chat template

**`max_seq_length = 2048`**

- Maximum tokens per example
- Longer sequences truncate ho jayengi
- Consistent with model loading

---

**TrainingArguments - Hyperparameters:**

---

**`per_device_train_batch_size=2`**

**Batch size kya hai?**

- Kitne examples **ek saath** train karenge

**Process:**

1. 2 examples select karo
2. Forward pass - dono ke predictions
3. Loss calculate karo - average of both
4. Backward pass - gradients compute karo

**Why 2?**

- Memory constraints
- 3B model with 2048 seq length
- Larger batch = more memory
- Smaller batch = more gradient noise but manageable

**Different sizes:**

- `1` - minimum, most memory efficient, noisy gradients
- `2` ✅ - good balance for this GPU
- `4` - better gradients but may OOM (Out of Memory)
- `8+` - better but needs more VRAM

---

**`gradient_accumulation_steps=4`**

**Ye kya hai? Bahut clever technique!**

**Problem:** Batch size 2 is small, gradients noisy hain

**Solution:** Simulate larger batch size without using more memory!

**Kaise?**

1. Forward pass on batch 1 (size 2)
2. Calculate gradients, **but don't update weights yet**
3. Forward pass on batch 2 (size 2)
4. **Accumulate** gradients
5. Forward pass on batch 3 (size 2)
6. Accumulate gradients
7. Forward pass on batch 4 (size 2)
8. Accumulate gradients
9. **Now** update weights with averaged gradients

**Effective batch size = per_device_batch_size × accumulation_steps**
= 2 × 4 = **8**

**Benefits:**

- Larger effective batch size
- More stable gradients
- No extra memory needed!
- Training more stable

**Trade-off:**

- Slower training (4x forward passes per update)
- But better quality updates

---

**`warmup_steps=5`**

**Learning rate warmup kya hai?**

**Problem:** Training start mein model unstable hota hai

- Random gradients
- Big updates can destabilize

**Solution:** Learning rate ko **gradually increase** karo

**Process:**

- Step 1: lr = 0.0 → 0.00004 (20% of target)
- Step 2: lr = 0.00004 → 0.00008 (40%)  
- Step 3: lr = 0.00008 → 0.00012 (60%)
- Step 4: lr = 0.00012 → 0.00016 (80%)
- Step 5: lr = 0.00016 → 0.0002 (100% - target)
- Step 6+: lr = 0.0002 (constant or decay)

**Benefits:**

- Prevents early instability
- Smoother training start
- Better convergence

**Why only 5?**

- Short training (60 steps)
- Warmup should be ~5-10% of total
- 5/60 = 8.3%

---

**`max_steps=60`**

**Training length:**

- Total **60 optimization steps**
- This is **very short** - just for demo!

**What's one step?**

- One weight update
- Remember: gradient_accumulation_steps=4
- So 60 steps = 60 × 4 = 240 forward passes
- Effective data seen = 240 × 2 (batch size) = **480 examples**

**Production mein:**

- Zyada steps use karo
- Typically: 500-5000 steps depending on dataset
- Monitor loss - jab plateau ho, stop karo
- Use validation set to avoid overfitting

**Alternative: `num_train_epochs`**

```python
num_train_epochs=3  # Complete dataset 3 times
```

- Epoch = full pass through dataset
- 100k examples / batch size 2 / accum 4 = 12,500 steps per epoch
- 3 epochs = 37,500 steps

**For learning:** 60 steps enough
**For production:** 1000+ steps recommended

---

**`learning_rate=2e-4`**

**Sabse critical hyperparameter!**

**Learning rate kya hai?**

- Har step mein weights kitna change karein
- Formula: `new_weight = old_weight - (learning_rate × gradient)`

**Value: 2e-4 = 0.0002**

**Too high (e.g., 0.01):**

- Fast learning
- But: Overshooting
- Loss may increase
- Training unstable
- Model may diverge

**Too low (e.g., 0.00001):**

- Very stable
- But: Extremely slow learning
- May not converge in reasonable time
- Stuck in local minima

**2e-4 is perfect for LoRA because:**

- LoRA adapters are small
- Need relatively higher LR than full fine-tuning
- Well-tested value
- Good convergence speed

**Other common values:**

- Full fine-tuning: 1e-5 to 5e-5 (lower)
- LoRA: 1e-4 to 3e-4 (higher) ✅
- QLoRA: 2e-4 to 5e-4

**Best practice:** Start with 2e-4, monitor loss

- Loss decreasing? ✅ Good
- Loss unstable? → Decrease to 1e-4
- Loss not moving? → Increase to 3e-4

---

**`fp16=not torch.cuda.is_bf16_supported()`**
**`bf16=torch.cuda.is_bf16_supported()`**

**Mixed precision training! Memory aur speed boost!**

**Precision types:**

**FP32 (Float32 - Full Precision):**

- 32 bits per number
- Range: ±3.4 × 10^38
- Precision: ~7 decimal digits
- Memory: Most
- Speed: Slowest

**FP16 (Float16 - Half Precision):**

- 16 bits per number
- Range: ±65,504 (much smaller!)
- Precision: ~3 decimal digits
- Memory: 50% less
- Speed: ~2-3x faster
- Problem: **Numerical instability** - very small/large numbers overflow

**BF16 (Brain Float16):**

- 16 bits per number
- Range: Same as FP32! ±3.4 × 10^38
- Precision: Less than FP16
- Memory: 50% less
- Speed: ~2-3x faster
- **No overflow issues!** ✅

**BF16 vs FP16:**

```
FP32: [sign: 1 bit][exponent: 8 bits][mantissa: 23 bits]
FP16: [sign: 1 bit][exponent: 5 bits][mantissa: 10 bits] ← Limited range
BF16: [sign: 1 bit][exponent: 8 bits][mantissa: 7 bits]  ← Same range as FP32!
```

**GPU support:**

- Older GPUs (T4, V100): Only FP16
- Newer GPUs (A100, H100, 4090): BF16 ✅ Better
- Colab T4: FP16

**Code logic:**

```python
fp16 = not torch.cuda.is_bf16_supported()  # True if BF16 not available
bf16 = torch.cuda.is_bf16_supported()      # True if BF16 available
```

**Result:**

- BF16 available? → Use BF16 (better)
- BF16 not available? → Use FP16 (fallback)
- **Always use some mixed precision** - huge benefits!

**Benefits:**

- 50% less memory
- 2-3x faster training
- Minimal accuracy loss
- Industry standard

---

**`logging_steps=1`**

**Training progress tracking:**

**Kya log hota hai?**

- Loss value - kitna error hai
- Learning rate - current LR
- Step number
- Training speed (samples/second)
- Time elapsed

**`logging_steps=1` means:**

- **Har step** ke baad log karo
- Bahut detailed monitoring
- Useful for:
  - Debugging
  - Understanding training dynamics
  - Spotting issues early

**Production mein:**

- `logging_steps=10` or `logging_steps=50`
- Kam frequent logging
- Less console clutter

**Example output:**

```
Step 1: loss=2.456, lr=0.00004
Step 2: loss=2.234, lr=0.00008
Step 3: loss=2.123, lr=0.00012
...
Step 60: loss=0.567, lr=0.0002
```

**Loss ko track karna zaroori:**

- Decreasing? ✅ Model learning kar raha hai
- Increasing? ❌ Learning rate too high
- Flat? Learning rate too low ya converged

---

**`output_dir="outputs"`**

**Kya save hoga?**

- Checkpoints - model weights at different steps
- Training logs
- Optimizer state
- Trainer state
- Configuration files

**Directory structure:**

```
outputs/
├── checkpoint-500/
│   ├── adapter_model.safetensors  ← LoRA weights
│   ├── adapter_config.json
│   ├── optimizer.pt
│   └── trainer_state.json
├── checkpoint-1000/
│   └── ...
└── runs/  ← TensorBoard logs (if enabled)
```

**Best practice:**

- Descriptive names: `"outputs/llama3_finance_finetuned"`
- Include date: `"outputs/2024_01_15_experiment"`
- Version control exclude karo (.gitignore)

---

**Advanced arguments (not used but good to know):**

```python
TrainingArguments(
    # Optimizer
    optim="adamw_torch",  # Default: AdamW optimizer
    weight_decay=0.01,    # L2 regularization
    adam_beta1=0.9,       # Adam parameter
    adam_beta2=0.999,     # Adam parameter
    
    # Learning rate schedule
    lr_scheduler_type="cosine",  # Cosine decay
    
    # Saving
    save_steps=500,              # Checkpoint frequency
    save_total_limit=3,          # Keep only 3 checkpoints
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    
    # Other
    seed=42,                     # Reproducibility
    dataloader_num_workers=4,    # Parallel data loading
    remove_unused_columns=True,  # Clean dataset
)
```

---

## PART 13: TRAINING EXECUTION (52:00 - 55:00)

### Cell 14: Starting Training

```python
trainer.train()
```

**Ye single line kya karta hai? BAHUT KUCH!**

**Behind the scenes:**

**1. Initialization:**

- Optimizer setup (AdamW)
- Learning rate scheduler setup
- Loss function setup (CrossEntropyLoss for next token prediction)

**2. Training loop begins:**

```
For each step (1 to 60):
    For each accumulation step (1 to 4):
        1. Load batch (2 examples)
        2. Tokenize text
        3. Forward pass through model
        4. Compute loss
        5. Backward pass (compute gradients)
        6. Accumulate gradients
    
    7. Optimizer step (update weights)
    8. Zero gradients
    9. Log metrics
    10. Learning rate schedule step
```

**3. What happens during forward pass:**

- Input tokens → Embeddings
- Pass through 28 transformer layers
- Each layer:
  - Attention mechanism (with LoRA adapters)
  - Feed-forward network (with LoRA adapters)
- Output → Logits (probabilities for next token)

**4. Loss calculation:**

- Compare predicted next token with actual
- CrossEntropyLoss
- Average across batch

**5. Backward pass:**

- Gradient computation
- Only LoRA adapters get gradients (99% frozen!)
- Memory efficient

**6. Weight update:**

- AdamW optimizer
- Update LoRA matrices A and B
- Original weights untouched

---

**Training output:**

```
{'loss': 2.4561, 'learning_rate': 4e-05, 'epoch': 0.0}
{'loss': 2.1234, 'learning_rate': 8e-05, 'epoch': 0.01}
{'loss': 1.8923, 'learning_rate': 0.00012, 'epoch': 0.01}
...
{'loss': 0.5431, 'learning_rate': 0.0002, 'epoch': 0.05}
{'train_runtime': 180.5, 'train_samples_per_second': 5.32}
```

**What to monitor:**

**Loss decreasing:** ✅ Good

- Start: ~2.5
- End: ~0.5
- Shows learning happening

**Learning rate:**

- Warmup: 0 → 2e-4
- Then constant

**Epoch:**

- 60 steps / 100k dataset = 0.06 epoch
- Very small fraction of data

---

**Training time:**

- T4 GPU: ~3-5 minutes for 60 steps
- Real training (1000 steps): ~50-80 minutes
- Production (5000 steps): ~4-6 hours

**Memory usage:**

- Peak: ~6-8 GB
- Comfortable on 15GB GPU

**If OOM (Out of Memory) error:**

- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 1024
- Use smaller model (1B)

---

## PART 14: MODEL SAVING (55:00 - 56:30)

### Cell 15: Saving Fine-tuned Model

```python
model.save_pretrained("finetuned_model")
```

**Kya save ho raha hai?**

**NOT the full model! Only LoRA adapters! ✅**

**Files saved:**

```
finetuned_model/
├── adapter_config.json       ← LoRA configuration
├── adapter_model.safetensors ← LoRA weights (~20-50 MB)
└── README.md                 ← Auto-generated info
```

**Why only adapters?**

- Base model already Hugging Face pe hai
- Adapters hi train kiye hain
- 3B model: ~6GB
- Adapters: ~30MB
- **200x smaller!** ✅

**adapter_config.json:**

```json
{
  "r": 16,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "k_proj", ...],
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

**adapter_model.safetensors:**

- Actual LoRA weight matrices
- A and B matrices for each targeted layer
- Binary format, efficient

**Best practice:**

- Descriptive folder name
- Include training date/version
- Save config separately
- Share on Hugging Face Hub (optional)

**Upload to Hub:**

```python
model.push_to_hub("username/my-finetuned-llama")
```

---

## PART 15: INFERENCE (56:30 - 62:00)

### Cell 16: Loading Fine-tuned Model

```python
inference_model, inference_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)
```

**Local model loading:**

**`model_name="./finetuned_model"`:**

- `./` indicates **local path**
- Loading from disk, not Hugging Face Hub
- Faster load time

**Behind the scenes:**

1. Load base model: `unsloth/Llama-3.2-3B-Instruct`
2. Load LoRA adapters from `./finetuned_model`
3. Merge them (logically, not physically)
4. Ready for inference!

**Same parameters:**

- `max_seq_length=2048` - consistency
- `load_in_4bit=True` - memory efficiency

---

### Cell 17: Testing the Model

```python
text_prompts = [
    "What are the key principles of investment?"
]
```

**Test prompts:**

- Financial question
- Checks if model learned from training data
- Can add multiple prompts for testing

---

```python
for prompt in text_prompts:
```

**Loop through prompts** - testing multiple ek saath

---

```python
formatted_prompt = inference_tokenizer.apply_chat_template([{
    "role": "user",
    "content": prompt
}], tokenize=False)
```

**Chat formatting:**

**Input:**

```python
[{"role": "user", "content": "What are the key principles of investment?"}]
```

**Output (formatted_prompt):**

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What are the key principles of investment?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

**Important:**

- Ends with assistant header
- Model will continue from here
- Proper format ensures good response

---

```python
model_inputs = inference_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
```

**Tokenization for inference:**

**`inference_tokenizer(formatted_prompt, ...)`:**

- Converts text → token IDs
- Example: "What" → 3555, "are" → 527, ...

**`return_tensors="pt"`:**

- Return as **PyTorch tensors**
- Not Python lists
- Required for model input

**`.to("cuda")`:**

- Move tensors to GPU
- Faster inference
- Model is already on GPU

**Result (model_inputs):**

```python
{
  'input_ids': tensor([[1, 128000, 128006, 882, ...]]),      # Token IDs
  'attention_mask': tensor([[1, 1, 1, 1, ...]])               # Which tokens to attend
}
```

---

```python
generated_ids = inference_model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=inference_tokenizer.pad_token_id
)
```

**Generation parameters - bahut important!**

---

**`**model_inputs`:**

- Unpacks dictionary
- Equivalent to: `input_ids=..., attention_mask=...`

---

**`max_new_tokens=512`:**

**Kya hai?**

- Maximum tokens in **response**
- Not total (input + output), only output

**Example:**

- Input: 50 tokens
- max_new_tokens: 512
- Maximum output: 512 tokens
- Total: 562 tokens

**Why 512?**

- Good length for detailed answers
- Not too long (saves time)
- Won't hit 2048 limit easily

**Adjust karna:**

- Short answers: 128-256
- Medium: 512 ✅
- Long: 1024
- Very long: 2048

---

**`temperature=0.7`:**

**Ye creativity control karta hai! Super important!**

**How it works:**

Model generates **probability distribution** over vocabulary:

```
Token "the": 0.35
Token "a": 0.20
Token "investment": 0.15
Token "an": 0.10
...
```

**Temperature adjusts these probabilities:**

**temp = 1.0** (default):

- Use probabilities as-is
- Balanced randomness

**temp < 1.0** (e.g., 0.1):

- Make high probabilities higher
- Make low probabilities lower
- More deterministic
- Less creative
- Example: "the, the, the" → repetitive
- Use for: Factual answers, code, math

**temp > 1.0** (e.g., 1.5):

- Flatten distribution
- More randomness
- More creative
- More diverse
- Use for: Creative writing, brainstorming

**0.7 is sweet spot:** ✅

- Slightly more deterministic than 1.0
- Good balance
- Coherent + some diversity
- Best for: Instructions, Q&A, general chat

**Extreme values:**

- temp = 0: Always pick highest (deterministic)
- temp = 2.0: Very random (often incoherent)

---

**`do_sample=True`:**

**Sampling strategy:**

**do_sample=False (Greedy):**

- Always pick **highest probability** token
- Deterministic
- No creativity
- Can get repetitive

**do_sample=True (Sampling):** ✅

- **Sample** from probability distribution
- Temperature controls how
- More diverse outputs
- Better quality

**Best practice:**

- For chat/instructions: do_sample=True ✅
- For exact tasks (code, math): do_sample=False

---

**`pad_token_id=inference_tokenizer.pad_token_id`:**

**Padding kya hai?**

**Problem:** Batched generation mein sequences different length hote hain

**Example:**

```
Sequence 1: [1, 2, 3, 4, 5]        # Length 5
Sequence 2: [1, 2, 3]              # Length 3
```

**Solution:** Short sequences ko padd karo

```
Sequence 1: [1, 2, 3, 4, 5]        # Length 5
Sequence 2: [1, 2, 3, 0, 0]        # Length 5 (0 = pad token)
```

**Why specify pad_token_id?**

- Llama models don't have default pad token
- Explicitly set karna padta hai
- Prevents warnings/errors

**For single generation:** Not critical but good practice

---

**Other useful parameters (not used but good to know):**

```python
generate(
    # Length control
    min_new_tokens=10,        # Minimum response length
    max_new_tokens=512,
    
    # Sampling control  
    temperature=0.7,
    top_p=0.9,                # Nucleus sampling (alternative to temp)
    top_k=50,                 # Top-k sampling
    do_sample=True,
    
    # Repetition control
    repetition_penalty=1.2,   # Discourage repetition
    no_repeat_ngram_size=3,   # Don't repeat 3-grams
    
    # Stopping
    eos_token_id=tokenizer.eos_token_id,  # Stop token
    pad_token_id=tokenizer.pad_token_id,
    
    # Beam search (alternative to sampling)
    num_beams=4,              # Beam search (set do_sample=False)
    
    # Other
    use_cache=True,           # KV cache for speed
)
```

---

```python
response = inference_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**Decoding output:**

**`batch_decode()`:**

- Token IDs → Text
- Can handle multiple sequences (batch)
- Returns list of strings

**`generated_ids`:**

- Tensor of token IDs
- Includes input + generated tokens
- Shape: [batch_size, sequence_length]

**`skip_special_tokens=True`:**

- Remove special tokens from output
- No `<|begin_of_text|>`, `<|eot_id|>` etc.
- Clean readable text

**`[0]`:**

- First (and only) item in batch
- Single string extracted

**Result:**
Full conversation including prompt and response

---

```python
print(response)
```

**Output example:**

```
What are the key principles of investment?

The key principles of investment include:

1. Diversification: Spread your investments across different asset classes...
2. Risk Management: Understand your risk tolerance...
3. Long-term Perspective: Investing is a marathon, not a sprint...
4. Dollar-Cost Averaging: Invest fixed amounts regularly...
5. Research and Due Diligence: Always research before investing...

These principles help build a solid investment strategy and minimize risks.
```

---

**Response quality check:**

**Good signs:** ✅

- Relevant answer
- Coherent structure
- Follows instructions
- Uses training format

**Bad signs:** ❌

- Repetitive text
- Gibberish
- Off-topic
- Ignores instruction

**If quality poor:**

- Train longer (more steps)
- Better quality dataset
- Adjust learning rate
- Check if base model appropriate

---

## CONCLUSION (62:00 - 65:00)

**Toh dosto, ye tha complete fine-tuning guide!**

**Aaj humne kya seekha?**

✅ **Fine-tuning concepts:**

- Supervised fine-tuning
- LoRA and QLoRA
- PEFT techniques

✅ **Practical implementation:**

- Library installation and setup
- Model and tokenizer loading
- 4-bit quantization
- LoRA adapter configuration
- Dataset preparation
- Chat template formatting
- Training configuration
- Model training
- Inference and testing

✅ **Deep understanding:**

- Har library ka purpose
- Har parameter ka meaning
- Har line of code explained
- Best practices

---

**Key takeaways:**

1. **Always use QLoRA** for resource-constrained environments
2. **r=16** is the sweet spot for LoRA rank
3. **Target important modules:** attention + FFN
4. **Chat templates are crucial** for proper formatting
5. **Start with good base models** that have relevant knowledge
6. **Monitor loss** during training
7. **Save only adapters** - efficient storage
8. **Test thoroughly** after training

---

**Next steps for you:**

📚 **Experiment:**

- Try different datasets
- Adjust hyperparameters
- Different model sizes (1B, 8B)
- Longer training

🚀 **Production:**

- More training steps (1000+)
- Validation set for evaluation
- Proper testing suite
- Deploy using vLLM/TGI

💡 **Advanced topics (future videos):**

- DPO (Direct Preference Optimization)
- Multi-GPU training
- Full fine-tuning strategies
- Custom dataset creation
- Evaluation metrics
- Deployment strategies

---

**Resources:**

🔗 **Code:** (GitHub link daal dena)
🔗 **Dataset:** Hugging Face - mlabonne/FineTome-100k
🔗 **Model:** unsloth/Llama-3.2-3B-Instruct
📄 **Paper:** Attention is All You Need
📚 **Library Docs:** Unsloth, Transformers, TRL

---

**Final message:**

Dosto, fine-tuning ek **powerful skill** hai. Fortune 500 companies use karti hain, startups bana rahe hain is pe. Aap bhi master kar sakte ho!

**Practice karo, experiment karo, build karo!**

Agar video pasand aaya toh:

- 👍 Like karo
- 🔔 Subscribe karo visa2code channel ko
- 💬 Comments mein batao kaunse topics chahiye
- 🔗 Share karo apne friends ke saath

**Aur haan, code zaroor try karo! Theory se sirf samajh aati hai, coding se skills banti hain!**

**Next video mein milte hain with more exciting AI content!**

**Keep coding, keep learning!**

**This is visa2code, signing off! 🚀**

---

## BONUS SECTION: COMMON ISSUES & SOLUTIONS

### Issue 1: Out of Memory (OOM)

**Error:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Option 1: Reduce batch size
per_device_train_batch_size=1

# Option 2: Reduce sequence length
max_seq_length=1024

# Option 3: Increase gradient accumulation
gradient_accumulation_steps=8

# Option 4: Use smaller model
model_name="unsloth/Llama-3.2-1B-Instruct"
```

---

### Issue 2: Slow Training

**Solutions:**

```python
# Enable bf16/fp16
bf16=True  # If supported

# Reduce logging
logging_steps=10

# Optimize data loading
dataloader_num_workers=4

# Use Flash Attention (automatic in Unsloth)
```

---

### Issue 3: Poor Quality Outputs

**Solutions:**

- Train longer (more steps)
- Use better dataset
- Check chat template formatting
- Adjust learning rate
- Increase LoRA rank to 32

---

### Issue 4: Model Not Following Instructions

**Cause:** Chat template issue

**Solution:**

```python
# Ensure correct template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Verify formatting
print(dataset[0]['text'])  # Check output
```

---

### Issue 5: Installation Errors

**Solutions:**

```bash
# Update pip
pip install --upgrade pip

# Install with specific versions
pip install torch==2.1.0 transformers==4.36.0

# Clear cache
pip cache purge

# Restart runtime
```

---

**Ye complete guide hai! Har doubt clear hona chahiye! All the best! 🎉**
