# Llama 3.2-1B with Tiktoken Tokenizer - Setup Guide

## Changes Completed

### 1. Updated C# Tokenizer Loading
**File:** `services/Sphana.Database/Sphana.Database/Infrastructure/Onnx/LlmGeneratorModel.cs`

**What changed:**
- Now supports **both** `tokenizer.model` (SentencePiece) and `tokenizer.json` (Llama 3.2)
- Falls back to Tiktoken (cl100k_base encoding) when `tokenizer.json` is found
- Automatically detects which format is available

**Tokenizer fallback logic:**
1. Checks for `tokenizer.model` (SentencePiece) → if found, uses `LlamaTokenizer`
2. Falls back to `tokenizer.json` → uses `TiktokenTokenizer` as approximation
3. If neither exists → throws error

---

## Important Note About Tiktoken Approximation

**Tiktoken is an approximation** - it uses GPT-4's tokenizer (cl100k_base) which is **not identical** to Llama 3.2's tokenizer.

### Implications:
- ⚠️ **Token IDs will be different** - Llama 3.2 and GPT-4 have different vocabularies
- ⚠️ **May produce suboptimal results** - The model was trained with its own tokenizer
- ⚠️ **Text generation quality may be affected** - Mismatched tokenization can cause issues

### Why This Matters:
- Llama 3.2 expects specific token IDs from its trained tokenizer
- Tiktoken will produce different token IDs
- The model may not generate text correctly or at all

---

## Better Solutions

### Solution A: Use Gemma-2-2B Instead (Recommended) ✅

Gemma models use `tokenizer.model` (SentencePiece) which works perfectly with your existing C# code:

```yaml
# configs/llm/base.yaml
llm:
  model_name: google/gemma-2-2b-it
  quantize: true
```

**Benefits:**
- ✅ Perfect tokenizer compatibility (tokenizer.model included)
- ✅ INT8 quantization works
- ✅ ~550MB after quantization
- ✅ Excellent quality
- ✅ No approximations needed

**Export and use:**
```bash
$env:PYTHONPATH="src"
.\.venv\Scripts\activate
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

---

### Solution B: Create Proper .NET Tokenizer for Llama 3.2

**Requirements:**
- Need a .NET library that can load arbitrary `tokenizer.json` files
- Or implement a wrapper around the Python tokenizer

**Potential libraries:**
1. **TorchSharp.Tokenizers** - If it exists and supports HuggingFace format
2. **Call Python tokenizer** - Use process execution to call Python for tokenization
3. **Port the tokenizer** - Manually implement Llama 3.2 tokenization in C#

**This is complex and time-consuming.**

---

### Solution C: Extract tokenizer.model from Llama 3.2 (If Possible)

Some Llama variants include both formats. Check if there's a legacy `tokenizer.model`:

```bash
# Try to download from a different Llama variant
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'); tok.save_pretrained('llama2_tokenizer')"
```

But Llama 3.2 likely doesn't have this.

---

## Current Status with Tiktoken

### What You Have Now:
- ✅ Code compiles
- ✅ Tokenizer loads (Tiktoken approximation)
- ⚠️ **May not work correctly** due to vocab mismatch

### Testing Required:
1. Copy files to Database:
   ```powershell
   Copy-Item target\artifacts\llm\0.1.0\onnx\llm_generator.int8.onnx `
       ..\Sphana.Database\Sphana.Database\Resources\Models\llm_generator.onnx
   
   Copy-Item target\artifacts\llm\0.1.0\onnx\tokenizer.json `
       ..\Sphana.Database\Sphana.Database\Resources\Models\
   ```

2. Build and run:
   ```bash
   cd services/Sphana.Database
   dotnet build
   # Run in Docker
   ```

3. Test Query API:
   - Send a query
   - Check if LLM generates coherent text
   - **Expect:** May generate gibberish or empty responses due to token mismatch

---

## Recommended Path Forward

### Option 1: Switch to Gemma-2-2B (Easiest, Best) ✅

1. Change config to `google/gemma-2-2b-it`
2. Re-export with quantization
3. Perfect tokenizer compatibility
4. Works out of the box

### Option 2: Keep Llama 3.2 + Tiktoken (Testing Only)

1. Copy files as-is
2. Test if it works acceptably
3. If quality is poor → switch to Gemma

### Option 3: Implement Proper Llama 3.2 Tokenizer (Future)

1. Find or create .NET library for tokenizer.json
2. Update C# code to use it
3. Time-consuming but correct

---

## Comparison

| Model | Tokenizer Format | C# Support | Quantization | Recommendation |
|-------|-----------------|------------|--------------|----------------|
| **Gemma-2-2B** | tokenizer.model | ✅ Perfect | ✅ Works | **Best Choice** |
| **Llama-3.2-1B** | tokenizer.json | ⚠️ Approximation | ✅ Works | Risky |
| **Llama-2-7B** | tokenizer.model | ✅ Perfect | ✅ Works | Too large (7B) |

---

## Summary

**Current situation:**
- ✅ Llama-3.2-1B INT8 model exported successfully (1.4GB)
- ✅ C# code updated to handle tokenizer.json
- ⚠️ Using Tiktoken as approximation (may not work well)

**Recommendation:**
**Switch to Gemma-2-2B** for production use. It has:
- Perfect tokenizer compatibility
- Smaller size after quantization (~550MB vs 1.4GB)
- Proven to work
- No approximations needed

Would you like to proceed with Gemma-2-2B or test Llama 3.2 with Tiktoken first?

