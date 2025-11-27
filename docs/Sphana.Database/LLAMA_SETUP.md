# Llama-3.2-1B Setup Guide

## Changes Completed ✓

1. ✅ Created `quantize_llama_manual.py` - Manual INT8 quantization script
2. ✅ Updated `configs/llm/base.yaml` - Set to use `meta-llama/Llama-3.2-1B`
3. ✅ Updated `LlmGeneratorModel.cs` - Changed BOS/EOS token IDs to 128000/128001 for Llama

## Next Steps

### Step 1: Authenticate with HuggingFace (Required)

Llama models require accepting Meta's license:

```bash
# 1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B
# 2. Click "Agree and access repository"
# 3. Create a HuggingFace token: https://huggingface.co/settings/tokens
# 4. Login with the token:
huggingface-cli login
# Paste your token when prompted
```

### Step 2: Export Llama-3.2-1B to ONNX

```bash
cd services/Sphana.Trainer

# Run the standard export (quantization will fail, that's expected)
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

**Expected warnings (safe to ignore):**
- ✅ "The maximum absolute difference... is not within the set tolerance" - This is fine
- ⚠️ "Quantization with external data failed" - This is expected, we'll fix it next
- ℹ️ "Required inputs ['position_ids', 'past_key_values...] are missing" - This is normal validation behavior

**Expected output:**
- `target/artifacts/llm/0.1.0/onnx/llm_generator.onnx` (FP32 model)
- `target/artifacts/llm/0.1.0/onnx/model.onnx_data` (FP32 weights, ~4.8GB)
- `target/artifacts/llm/0.1.0/onnx/tokenizer.model` (SentencePiece tokenizer)
- `target/artifacts/llm/0.1.0/onnx/tokenizer.json` (HuggingFace format)

### Step 3: Run Manual INT8 Quantization

```bash
cd services/Sphana.Trainer

# Run the manual quantization script
python quantize_llama_manual.py
```

**Possible outcomes:**

#### A) Quantization Succeeds (Best Case) ✅
```
✓ Quantization succeeded!
INT8 model saved to: target/artifacts/llm/0.1.0/onnx/llm_generator.int8.onnx
INT8 model size: 1.20 GB
```

**Next:** Proceed to Step 4 with INT8 model

#### B) Quantization Fails (Fallback to FP32) ⚠️
```
✗ Method 1 failed...
✗ Method 2 failed...
✗ Method 3 failed...
All quantization methods failed.
You'll need to use the FP32 model.
```

**Next:** Proceed to Step 4 with FP32 model (requires 8GB+ VRAM)

### Step 4: Copy Files to Database Project

**If INT8 quantization succeeded:**

```bash
cd services/Sphana.Trainer

# Copy INT8 model
cp target/artifacts/llm/0.1.0/onnx/llm_generator.int8.onnx \
   ../Sphana.Database/Sphana.Database/Resources/Models/llm_generator.onnx

# Copy tokenizer
cp target/artifacts/llm/0.1.0/onnx/tokenizer.model \
   ../Sphana.Database/Sphana.Database/Resources/Models/

cp target/artifacts/llm/0.1.0/onnx/tokenizer.json \
   ../Sphana.Database/Sphana.Database/Resources/Models/
```

**If INT8 quantization failed (use FP32):**

```bash
cd services/Sphana.Trainer

# Copy FP32 model
cp target/artifacts/llm/0.1.0/onnx/llm_generator.onnx \
   ../Sphana.Database/Sphana.Database/Resources/Models/

cp target/artifacts/llm/0.1.0/onnx/model.onnx_data \
   ../Sphana.Database/Sphana.Database/Resources/Models/

# Copy tokenizer (same as INT8)
cp target/artifacts/llm/0.1.0/onnx/tokenizer.model \
   ../Sphana.Database/Sphana.Database/Resources/Models/

cp target/artifacts/llm/0.1.0/onnx/tokenizer.json \
   ../Sphana.Database/Sphana.Database/Resources/Models/
```

### Step 5: Build and Test

```bash
cd services/Sphana.Database

# Build the C# application
dotnet build

# If successful, run in Docker from Visual Studio
# OR run manually:
# docker-compose up
```

**Check logs for successful loading:**
```
INFO: Loaded Llama tokenizer from .../tokenizer.model, BOS=128000, EOS=128001
INFO: ONNX model loaded successfully: .../llm_generator.onnx, GPU: True
INFO: Model has 16 transformer layers
```

### Step 6: Test Query API

Send a test query:
```json
{
  "tenant_id": "tnt1",
  "index_name": "idx1",
  "query": "What is include_named_queries_score used for?",
  "top_k": 5
}
```

**Verify:**
- LLM generates a coherent, non-empty response
- Response time is reasonable (~80-100ms for INT8, ~200-250ms for FP32)
- Check VRAM usage with `nvidia-smi`

## Expected Performance

### With INT8 Quantization (If Successful):
- **Model size:** ~1.2GB
- **VRAM usage:** ~1.2GB
- **Total system VRAM:** ~1.7GB (all 5 models)
- **Inference speed:** ~80-100ms per query
- **GPU requirement:** 2GB+ VRAM
- **Recommended:** 4GB+ VRAM

### With FP32 (If Quantization Failed):
- **Model size:** ~4.8GB
- **VRAM usage:** ~4.8GB
- **Total system VRAM:** ~5.25GB (all 5 models)
- **Inference speed:** ~200-250ms per query
- **GPU requirement:** 8GB+ VRAM
- **Recommended:** 12GB+ VRAM

## Troubleshooting

### Issue: HuggingFace 401 Unauthorized
**Solution:** Follow Step 1 to authenticate

### Issue: Empty or Gibberish LLM Output
**Check:**
1. Verify BOS/EOS IDs are 128000/128001 in `LlmGeneratorModel.cs`
2. Ensure `tokenizer.model` file exists in Resources/Models
3. Check logs for tokenizer loading errors

### Issue: CUDA Out of Memory
**Solutions:**
1. Use INT8 model (not FP32)
2. Set `PreloadModelsInParallel: false` in appsettings.json
3. Reduce `BatchSize` in appsettings.json
4. Upgrade GPU if using FP32 model

### Issue: All Quantization Methods Fail
**Options:**
1. Use FP32 model (requires 8GB+ VRAM)
2. Switch to Gemma-2-2B (guaranteed INT8 support, ~550MB)
   - Change config: `model_name: google/gemma-2-2b-it`
   - Update BOS/EOS IDs back to 2/1 in LlmGeneratorModel.cs
   - Re-export

## Alternative: Gemma-2-2B (Fallback Option)

If Llama quantization fails and you don't have 8GB+ VRAM, Gemma-2-2B is a proven alternative:

```yaml
# configs/llm/base.yaml
llm:
  model_name: google/gemma-2-2b-it
  quantize: true
```

**Benefits:**
- ✅ INT8 quantization guaranteed to work
- ✅ ~550MB (smaller than Llama-3.2-1B INT8)
- ✅ Excellent quality for 2B model
- ✅ No authentication required
- ⚠️ Need to change BOS/EOS back to 2/1 in C#

## Summary

1. Authenticate with HuggingFace
2. Export Llama-3.2-1B (warnings expected)
3. Run manual quantization script
4. Copy model files to Resources/Models
5. Build and test C# application
6. Verify LLM generates proper responses

**Success criteria:** LLM loads successfully and generates coherent responses to queries.

