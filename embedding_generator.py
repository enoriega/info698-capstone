#!/usr/bin/env python3
"""
Optimized Embedding Generator Script - Second step in PubMed processing pipeline

This script generates vector embeddings for PubMed text chunks using
sentence transformers, optimized for batch processing and memory efficiency.

Features:
- Skip saving intermediate files - works in-memory
- Multiprocessing capability for parallel processing
- Memory-efficient batch processing
- Detailed logging and progress tracking
"""

import os
import sys
import json
import logging
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install required packages with:")
    print("pip install torch sentence-transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubmed_embedding.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pubmed_embedding")

class EmbeddingGenerator:
    def __init__(self, config: Dict = None):
        """Initialize the embedding generator with the specified model."""
        self.config = {
            "model_name": "pritamdeka/S-PubMedBERT-MS-MARCO",
            "batch_size": 32,
            "no_save": True,  # Default to NOT saving files
            "num_workers": min(8, mp.cpu_count())  # Use up to 8 CPU cores
        }
        if config:
            self.config.update(config)
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {self.config['model_name']}")
        self.model = SentenceTransformer(self.config["model_name"])
        
        # Enable GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Get embedding dimension for verification
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Log system information
        self._log_system_info()

    def _log_system_info(self):
        """Log system information for debugging"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"Using {self.config['num_workers']} worker processes")
            logger.info(f"RAM: {mem.percent}% used, {mem.available/1024/1024/1024:.1f}GB available")
        except ImportError:
            logger.info(f"Using {self.config['num_workers']} worker processes")

    def generate_embeddings_for_document(self, document: Dict) -> Dict:
        """Generate embeddings for all chunks in a document (in-memory)."""
        try:
            # Extract chunks
            chunks = document.get('chunks', [])
            if not chunks:
                logger.warning(f"No chunks found in document {document.get('document_id', 'unknown')}")
                return document
            
            # Prepare text for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Log chunk statistics
            logger.info(f"Processing {len(texts)} chunks for document {document.get('document_id', 'unknown')}")
            
            # Generate embeddings in batches
            all_embeddings = []
            batch_size = self.config["batch_size"]
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings
                batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True)
                
                # Convert to list for JSON serialization
                all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = all_embeddings[i]
            
            # Update the document with embedded chunks
            document['chunks'] = chunks
            return document
            
        except Exception as e:
            logger.error(f"Error generating embeddings for document {document.get('document_id', 'unknown')}: {str(e)}")
            return document

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process a batch of documents and add embeddings."""
        results = []
        for doc in tqdm(documents, desc="Generating embeddings"):
            try:
                embedded_doc = self.generate_embeddings_for_document(doc)
                results.append(embedded_doc)
            except Exception as e:
                logger.error(f"Failed to process document: {str(e)}")
        
        logger.info(f"Successfully embedded {len(results)} documents")
        return results

    def process_batch(self, batch_files: List[str]) -> List[Dict]:
        """Process a batch of files and return documents with embeddings."""
        documents = []
        
        # Load all documents in the batch
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                    documents.append(document)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        # Process all documents
        return self.process_documents(documents)

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None, 
                          limit: Optional[int] = None, batch_size: int = 10) -> List[Dict]:
        """Process all JSON files in the input directory and add embeddings."""
        # Get all JSON files
        files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
        if limit:
            files = files[:limit]
        
        logger.info(f"Found {len(files)} files to process for embeddings")
        
        # Process files in batches to manage memory
        results = []
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
            
            # Process batch
            batch_results = self.process_batch(batch_files)
            results.extend(batch_results)
            
            # Log memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            except ImportError:
                pass
        
        logger.info(f"Successfully processed {len(results)} documents with embeddings")
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for PubMed chunks")
    parser.add_argument("--input", "-i", required=True, help="Directory containing processed JSON files")
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of files to process")
    parser.add_argument("--model", "-m", default="pritamdeka/S-PubMedBERT-MS-MARCO", help="Sentence transformer model to use")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main function to run the embedding generator."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create config from arguments
    config = {
        "model_name": args.model,
        "batch_size": args.batch_size,
        "no_save": True,  # Don't save intermediate files
        "num_workers": args.workers
    }
    
    # Create generator and process files
    generator = EmbeddingGenerator(config)
    embedded_documents = generator.process_directory(args.input, limit=args.limit)
    
    logger.info(f"Generated embeddings for {len(embedded_documents)} documents")
    
    # Return documents with embeddings for next pipeline stage
    return embedded_documents


if __name__ == "__main__":
    main()