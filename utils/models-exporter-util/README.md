# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\utils\models-exporter-util`
2. Activate the Virtual Environment:<br>
   `.\utils\models-exporter-util\.venv\Scripts\activate`

## Refresh Dependencies
```
uv pip install -e .\utils\models-exporter-util\ --refresh --reinstall 
```

## Run Locally

### Download Embediding Model
```
uv run python .\utils\models-exporter-util\ --model-name nomic-ai/nomic-embed-text-v1.5 --output-path .\services\sphana-processor-service\src\resources\models\embedding
```


-----------


# Model Exporter Utility

A generic utility for downloading and saving HuggingFace models locally for offline use in the Sphana RAG service.

## Overview

This utility downloads models from HuggingFace Hub and saves them to a local directory, allowing the service to run without requiring internet access for model downloads during initialization.

## Prerequisites

Install the utility and its dependencies:

### Using pip
```bash
pip install .
```

### Using uv (recommended for faster installation)
```bash
uv pip install .
```

## Usage

### Basic Syntax

```bash
python export_models.py --model-name <model-identifier> --output-path <destination-path>
```

### Arguments

- `--model-name` (required): HuggingFace model identifier (e.g., `nomic-ai/nomic-embed-text-v1.5`)
- `--output-path` (required): Local directory path where the model will be saved
- `--trust-remote-code` (optional): Trust remote code execution (required for some models)

## Examples

### Download Nomic Embedding Model

Download the nomic-embed-text-v1.5 model to the RAG service resources directory:

```bash
python export_models.py \
  --model-name nomic-ai/nomic-embed-text-v1.5 \
  --output-path ../../services/sphana-rag-service/src/resources/models/nomic-embed-text-v1.5 \
  --trust-remote-code
```

### Download Other Sentence-Transformer Models

Download any sentence-transformers compatible model:

```bash
# BERT-based model
python export_models.py \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --output-path ../../services/sphana-rag-service/src/resources/models/all-MiniLM-L6-v2

# Multilingual model
python export_models.py \
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --output-path ../../services/sphana-rag-service/src/resources/models/multilingual-MiniLM
```

### Download Custom Fine-Tuned Models

```bash
python export_models.py \
  --model-name your-org/your-custom-model \
  --output-path ../../services/sphana-rag-service/src/resources/models/custom-model
```

## Output

The script will:
1. Create the output directory if it doesn't exist
2. Download the tokenizer and save it to the specified path
3. Download the sentence-transformer model and save it to the same path
4. Display progress and completion status

Example output:
```
2024-02-04 16:20:00 - INFO - Starting download of model: nomic-ai/nomic-embed-text-v1.5
2024-02-04 16:20:00 - INFO - Output path: C:\...\models\nomic-embed-text-v1.5
2024-02-04 16:20:05 - INFO - Downloading tokenizer...
2024-02-04 16:20:08 - INFO - ✓ Tokenizer saved to: C:\...\models\nomic-embed-text-v1.5
2024-02-04 16:20:08 - INFO - Downloading sentence-transformer model...
2024-02-04 16:21:30 - INFO - ✓ Model saved to: C:\...\models\nomic-embed-text-v1.5
2024-02-04 16:21:30 - INFO - ✓ Successfully saved 15 files/directories
2024-02-04 16:21:30 - INFO - ============================================================
2024-02-04 16:21:30 - INFO - MODEL EXPORT COMPLETED SUCCESSFULLY
2024-02-04 16:21:30 - INFO - ============================================================
```

## Error Handling

The script provides clear error messages for common issues:

- **Missing dependencies**: Instructions to install requirements
- **Invalid model name**: HuggingFace API error messages
- **Permission errors**: File system access issues
- **Network errors**: Connection problems with HuggingFace Hub

## Integration with Sphana RAG Service

After downloading a model, update the `TextTokenizer` class in the RAG service to use the local path:

```python
# services/sphana-rag-service/src/python/sphana_rag/services/tokenizer/text_tokenizer.py
self._local_model_path = "services/sphana-rag-service/src/resources/models/nomic-embed-text-v1.5"
```

The tokenizer will automatically check for the local model before attempting to download from HuggingFace.

## Notes

- Downloaded models can be several GB in size
- Ensure you have sufficient disk space before downloading
- Models are saved with their full configuration and weights
- The `--trust-remote-code` flag is required for models like nomic-ai that use custom code

## Supported Model Types

This utility supports:
- Sentence-Transformers models
- HuggingFace Transformers models with AutoTokenizer support
- Custom fine-tuned models
- Models requiring remote code execution (with `--trust-remote-code` flag)

## Troubleshooting

### "Missing required dependencies" error
```bash
pip install .
# or with uv
uv pip install .
```

### "Permission denied" error
Ensure you have write permissions to the output directory.

### Model download fails
- Check your internet connection
- Verify the model name is correct on HuggingFace Hub
- Some models may require authentication - set `HF_TOKEN` environment variable

## Future Enhancements

Planned features:
- Batch download multiple models
- Resume interrupted downloads
- Model verification and checksums
- Progress bars for large downloads