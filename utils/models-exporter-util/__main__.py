#!/usr/bin/env python3
"""
Generic Model Exporter Utility

This script downloads and saves HuggingFace models locally for offline use.
Supports both tokenizers and sentence-transformer models.

Usage:
    python export_models.py --model-name <model-identifier> --output-path <destination-path>

Example:
    python export_models.py --model-name nomic-ai/nomic-embed-text-v1.5 --output-path ../../services/sphana-rag-service/src/resources/models/nomic-embed-text-v1.5
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_name: str, output_path: str, trust_remote_code: bool = False) -> bool:
    """
    Download a model from HuggingFace and save it locally.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'nomic-ai/nomic-embed-text-v1.5')
        output_path: Local directory path where the model will be saved
        trust_remote_code: Whether to trust remote code execution (required for some models)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import here to provide better error messages if dependencies are missing
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Starting download of model: {model_name}")
        logger.info(f"Output path: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(output_path)
        logger.info(f"✓ Tokenizer saved to: {output_path}")
        
        # Download sentence-transformer model
        logger.info("Downloading sentence-transformer model...")
        model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote_code
        )
        model.save(output_path)
        logger.info(f"✓ Model saved to: {output_path}")
        
        # Verify the model was saved correctly
        saved_files = list(output_dir.iterdir())
        logger.info(f"✓ Successfully saved {len(saved_files)} files/directories")
        
        logger.info("=" * 60)
        logger.info("MODEL EXPORT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Location: {output_path}")
        logger.info(f"Files: {len(saved_files)}")
        logger.info("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main function to parse arguments and execute model export."""
    parser = argparse.ArgumentParser(
        description='Download and save HuggingFace models for offline use',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download nomic embedding model
  python export_models.py \\
    --model-name nomic-ai/nomic-embed-text-v1.5 \\
    --output-path ../../services/sphana-rag-service/src/resources/models/nomic-embed-text-v1.5

  # Download with trust remote code
  python export_models.py \\
    --model-name some-org/custom-model \\
    --output-path ./models/custom-model \\
    --trust-remote-code
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., nomic-ai/nomic-embed-text-v1.5)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Local directory path where the model will be saved'
    )
    
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=True,
        help='Trust remote code execution (default: True, required for models like nomic-ai)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_name.strip():
        logger.error("Model name cannot be empty")
        sys.exit(1)
    
    if not args.output_path.strip():
        logger.error("Output path cannot be empty")
        sys.exit(1)
    
    # Convert to absolute path for clarity
    output_path = os.path.abspath(args.output_path)
    
    # Execute download
    success = download_model(
        model_name=args.model_name,
        output_path=output_path,
        trust_remote_code=args.trust_remote_code
    )
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()