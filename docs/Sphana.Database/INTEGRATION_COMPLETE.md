# Integration Complete: Multi-Method Quantization

## ✅ Summary of Changes

### 1. Updated `_quantize_with_external_data` Function
**File:** `services/Sphana.Trainer/src/sphana_trainer/exporters/onnx.py` (lines 422-497)

**What was changed:**
- Replaced single-attempt quantization with **3-method fallback strategy**
- Added proper error handling and cleanup for each method
- Added detailed logging for each attempt

**Methods implemented (in order):**
1. **Standard quantization** - Basic `quantize_dynamic()` call
2. **With extra options** - Uses `extra_options={'EnableSubgraph': False}` for complex models
3. **Embedded first** - Embeds external data before quantization (fallback for Llama-3.2)

### 2. Added Helper Function `_quantize_via_embedding`
**File:** `services/Sphana.Trainer/src/sphana_trainer/exporters/onnx.py` (lines 500-515)

**Purpose:**
- Handles the "embed external data first" quantization approach
- Used as the 3rd fallback method
- Properly cleans up temporary files

### 3. Fixed Potential Bugs
- **Variable initialization** - Ensures `embedded_path`, `quantized_embedded`, `preprocessed` are defined before cleanup
- **Null checks** - Added `if temp_file and temp_file.exists()` to prevent errors

---

## How It Works Now

### Before (Old Behavior):
```python
try:
    # Single quantization attempt
    quantize_dynamic(...)
    return target
except Exception:
    # Give up immediately
    return source
```

### After (New Behavior):
```python
methods = [method1, method2, method3]

for method in methods:
    try:
        # Try this method
        quantize_dynamic(...)
        return target  # Success!
    except Exception:
        # Try next method
        continue

# All methods failed
return source
```

---

## Testing Status

### ✅ Completed:
1. Code integration complete
2. Linting errors fixed
3. Syntax validated
4. Logic reviewed

### ⏳ To Test:
Run the standard export command to verify the multi-method fallback works:

```bash
cd services/Sphana.Trainer
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

**Expected behavior:**
- If Method 1 (standard) fails → automatically tries Method 2
- If Method 2 fails → automatically tries Method 3
- If all fail → warns and uses FP32 model
- Logs show which method succeeded

---

## Benefits of This Integration

### 1. **No Manual Script Needed**
- Before: Had to run `quantize_llama_manual.py` separately
- After: Automatic fallback built into `train llm` command

### 2. **Better User Experience**
- Automatically tries multiple approaches
- Clear logging shows which method worked
- No need to understand quantization internals

### 3. **Consistent with Other Models**
- Uses same `_quantize_or_fallback` pattern as embedding/relation models
- Integrated into existing pipeline

### 4. **Robust for Future Models**
- Will work for new model architectures with similar issues
- Fallback methods handle various ONNX structures

---

## Comparison: Manual Script vs Integrated

| Feature | Manual Script | Integrated Version |
|---------|--------------|-------------------|
| **Invocation** | Separate command | Automatic |
| **Methods tried** | 3 | 3 (same) |
| **Logging** | Basic print statements | Proper LOGGER.info/warning |
| **Error handling** | Try-catch per method | Try-catch per method |
| **Cleanup** | Manual | Automatic |
| **File size info** | Yes | No (not needed in pipeline) |
| **Integration** | Standalone | Part of export flow |

---

## What You Can Do Now

### Option 1: Delete Manual Script (Recommended)
Since the capabilities are now integrated, you can:
```bash
rm services/Sphana.Trainer/quantize_llama_manual.py
```

### Option 2: Keep as Reference
Keep the script as a standalone tool for manual testing/debugging.

---

## Next Steps for Testing

1. **Clean previous artifacts:**
```bash
cd services/Sphana.Trainer
rm -rf target/artifacts/llm/0.1.0
```

2. **Run export with integrated quantization:**
```bash
python -m sphana_trainer.cli train llm --config configs/llm/base.yaml
```

3. **Check logs for:**
- "Attempting quantization method: standard"
- "Attempting quantization method: with_extra_options"  
- "Attempting quantization method: embedded_first"
- "Successfully quantized model... using method: X"

4. **Verify INT8 model created:**
```bash
ls -lh target/artifacts/llm/0.1.0/onnx/*.int8.onnx
# Should show ~1.2-1.4GB file
```

---

## Current Status

✅ **COMPLETE** - Multi-method quantization is now integrated into the `train llm` command!

The `quantize_llama_manual.py` script served its purpose as a proof-of-concept, and its capabilities are now part of the main export pipeline.

