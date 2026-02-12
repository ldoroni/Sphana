import logging
import os
import sys
from pathlib import Path
from injector import singleton
from prometheus_client import Counter, Histogram
from time import time
from transformers import AutoTokenizer
from sphana_rag.models import TokenizedText

TOKENIZER_EXE_COUNTER = Counter("spn_tokenizer_exe_total", "Total number of tokenizer operations executed", ["operation"])
TOKENIZER_EXE_DURATION_HISTOGRAM = Histogram("spn_tokenizer_exe_duration_seconds", "Duration of tokenizer operations in seconds", ["operation"])

@singleton
class TextTokenizer:
    
    def __init__(self):
        self.__logger = logging.getLogger(self.__class__.__name__)
        
        # Calculate absolute path to local model
        # From: services/sphana-rag-service/src/python/sphana_rag/__main__.py
        # To:   services/sphana-rag-service/src/resources/models/embedding
        current_file = Path(sys.argv[0]).resolve()
        service_root = current_file.parents[2]
        local_model_path = str(service_root / "resources" / "models" / "embedding")
        
        # Verify local model exists
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model not found at: {local_model_path}")
        
        # Verify it's a valid model directory
        model_path = Path(local_model_path)
        required_files = ['config.json']
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Local model directory exists but is incomplete; Missing files: {missing_files}; Path: {local_model_path}")
        
        # Load the tokenizer
        self.__tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        
        self.__logger.info(f"TextTokenizer initialized with model {local_model_path}")

    def tokenize_text(self, text: str) -> TokenizedText:
        start_time: float = time()
        TOKENIZER_EXE_COUNTER.labels(operation="tokenize").inc()
        try:
            if not text or not text.strip():
                return TokenizedText(
                    text=text, 
                    token_ids=[], 
                    offsets=[]
                )
            
            encoding = self.__tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False
            )
            
            return TokenizedText(
                text=text,
                token_ids=encoding['input_ids'],
                offsets=encoding['offset_mapping']
            )
        finally:
            duration: float = time() - start_time
            TOKENIZER_EXE_DURATION_HISTOGRAM.labels(operation="tokenize").observe(duration)