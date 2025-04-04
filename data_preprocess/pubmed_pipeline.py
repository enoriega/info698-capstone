#!/usr/bin/env python3
"""
PubMed Complete Pipeline Script

This script runs the complete PubMed processing pipeline:
1. Parse and chunk PubMed XML files
2. Generate embeddings for chunks
3. Store in Weaviate for vector search

Usage:
    python pubmed_pipeline.py --input /path/to/xmls --limit 20

Environment variables:
    WEAVIATE_API_KEY - Required for Weaviate authentication
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import dotenv for loading API key from .env file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import from our modules
try:
    from pubmed_parser import PubMedProcessor
    from embedding_generator import EmbeddingGenerator
    from weaviate_manager import WeaviateManager
except ImportError:
    # If modules not installed, use local imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pubmed_parser import PubMedProcessor
    from embedding_generator import EmbeddingGenerator
    from weaviate_manager import WeaviateManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubmed_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pubmed_pipeline")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete PubMed processing pipeline")
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Directory containing XML files")
    
    # Optional arguments
    parser.add_argument("--output-dir", "-o", default="./processed", help="Output directory for processed files")
    parser.add_argument("--embeddings-dir", "-e", default="./processed_with_embeddings", help="Output directory for files with embeddings")
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of files to process")
    parser.add_argument("--chunk-size", type=int, default=350, help="Target chunk size in tokens")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate server URL")
    parser.add_argument("--weaviate-grpc-host", default="localhost", help="Weaviate gRPC host")
    parser.add_argument("--weaviate-grpc-port", type=int, default=50051, help="Weaviate gRPC port")
    parser.add_argument("--collection", default="PubMedArticle", help="Weaviate collection name")
    parser.add_argument("--model", default="pritamdeka/S-PubMedBERT-MS-MARCO", help="Embedding model name")
    parser.add_argument("--skip-weaviate", action="store_true", help="Skip loading to Weaviate")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()

def main():
    """Main function to run the complete pipeline."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check for required environment variables
    if not args.skip_weaviate and not os.environ.get("WEAVIATE_API_KEY"):
        logger.warning("WEAVIATE_API_KEY environment variable not set. Weaviate indexing may fail.")
    
    # Create configuration
    config = {
        "output_dir": args.output_dir,
        "embeddings_dir": args.embeddings_dir,
        "chunk_size": args.chunk_size,
        "weaviate_url": args.weaviate_url,
        "weaviate_grpc_host": args.weaviate_grpc_host,
        "weaviate_grpc_port": args.weaviate_grpc_port,
        "weaviate_api_key": os.environ.get("WEAVIATE_API_KEY"),
        "collection_name": args.collection,
        "embedding_model": args.model
    }
    
    logger.info("Starting PubMed processing pipeline")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit} files")
    
    # Step 1: Parse and chunk PubMed XML files
    logger.info("Step 1: Parsing and chunking PubMed XML files")
    processor = PubMedProcessor({
        "output_dir": args.output_dir,
        "chunk_size": args.chunk_size
    })
    processed_files = processor.process_directory(args.input, file_limit=args.limit)
    logger.info(f"Step 1 complete: Processed {len(processed_files)} files")
    
    # Step 2: Generate embeddings
    logger.info("Step 2: Generating embeddings for chunks")
    embedding_generator = EmbeddingGenerator({
        "model_name": args.model,
        "output_dir": args.embeddings_dir
    })
    embedding_files = embedding_generator.process_directory(args.output_dir, args.embeddings_dir, args.limit)
    logger.info(f"Step 2 complete: Generated embeddings for {len(embedding_files)} files")
    
    # Step 3: Store in Weaviate (optional)
    if not args.skip_weaviate:
        logger.info("Step 3: Storing data in Weaviate")
        weaviate_manager = WeaviateManager({
            "weaviate_url": args.weaviate_url,
            "weaviate_grpc_host": args.weaviate_grpc_host,
            "weaviate_grpc_port": args.weaviate_grpc_port,
            "weaviate_api_key": os.environ.get("WEAVIATE_API_KEY"),
            "collection_name": args.collection
        })
        
        # Check Weaviate connection
        if weaviate_manager.check_connection():
            indexed_count = weaviate_manager.index_directory(args.embeddings_dir, args.limit)
            logger.info(f"Step 3 complete: Indexed {indexed_count} files to Weaviate")
        else:
            logger.error("Failed to connect to Weaviate. Skipping indexing step.")
    else:
        logger.info("Step 3 skipped: Not storing data in Weaviate")
    
    logger.info("PubMed processing pipeline complete!")
    
    # Print summary
    print("\nPipeline Test Summary:")
    print("="*50)
    print(f"1. XML Processing: Processed {len(processed_files)} files")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Generated JSON files with chunked text")
    
    print(f"\n2. Embedding Generation: Generated embeddings for {len(embedding_files)} files")
    print(f"   - Output directory: {args.embeddings_dir}")
    print(f"   - Model: {args.model}")
    print(f"   - Embedding dimension: {embedding_generator.embedding_dim}")
    
    if not args.skip_weaviate:
        print(f"\n3. Weaviate Indexing: {'Complete' if weaviate_manager.check_connection() else 'Failed'}")
        print(f"   - Weaviate URL: {args.weaviate_url}")
        print(f"   - Collection: {args.collection}")
        
        # Try to get total count
        try:
            if weaviate_manager.check_connection():
                collection = weaviate_manager.client.collections.get(args.collection)
                count = collection.query.fetch_objects(limit=1, include_vector=False).total_count
                print(f"   - Total chunks indexed: {count}")
        except:
            print(f"   - Total chunks indexed: Unknown (couldn't query Weaviate)")
    
    print("\nPipeline complete!")
    
if __name__ == "__main__":
    main()