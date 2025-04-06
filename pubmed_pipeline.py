
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import psutil
import json

# Import from our modules
try:
    from pubmed_parser import PubMedProcessor
    from embedding_generator import EmbeddingGenerator
    from index_embedding import WeaviateManager
except ImportError:
    # If modules not installed, use local imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pubmed_parser import PubMedProcessor
    from embedding_generator import EmbeddingGenerator
    from index_embedding import WeaviateManager

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

class PipelineStats:
    """Track statistics for the pipeline."""
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.files_failed = 0
        self.chunks_generated = 0
        self.chunks_embedded = 0
        self.chunks_indexed = 0
        self.batches_completed = 0
        
    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
        
    def get_summary(self):
        """Get summary of statistics."""
        elapsed = self.get_elapsed_time()
        return {
            "elapsed_time": f"{elapsed:.2f} seconds",
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "chunks_generated": self.chunks_generated,
            "chunks_embedded": self.chunks_embedded,
            "chunks_indexed": self.chunks_indexed,
            "batches_completed": self.batches_completed,
            "files_per_second": f"{self.files_processed / elapsed:.2f}" if elapsed > 0 else "0",
            "chunks_per_second": f"{self.chunks_generated / elapsed:.2f}" if elapsed > 0 else "0"
        }
    
    def print_summary(self):
        """Print summary of statistics."""
        summary = self.get_summary()
        print("\nPipeline Summary:")
        print("="*50)
        print(f"Total time: {summary['elapsed_time']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Files failed: {summary['files_failed']}")
        print(f"Chunks generated: {summary['chunks_generated']}")
        print(f"Chunks embedded: {summary['chunks_embedded']}")
        print(f"Chunks indexed: {summary['chunks_indexed']}")
        print(f"Batches completed: {summary['batches_completed']}")
        print(f"Processing speed: {summary['files_per_second']} files/sec")
        print(f"                  {summary['chunks_per_second']} chunks/sec")

def get_system_info():
    """Get system resource information."""
    cpu_count = os.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu": {
            "total_cores": cpu_count,
            "usage_percent": cpu_percent
        },
        "memory": {
            "total_gb": f"{memory.total / (1024**3):.1f}",
            "available_gb": f"{memory.available / (1024**3):.1f}",
            "used_percent": memory.percent
        },
        "disk": {
            "total_gb": f"{disk.total / (1024**3):.1f}",
            "free_gb": f"{disk.free / (1024**3):.1f}",
            "used_percent": disk.percent
        }
    }

def print_system_info():
    """Print system resource information."""
    info = get_system_info()
    print("\nSystem Resources:")
    print(f"CPU: {info['cpu']['usage_percent']}% of {info['cpu']['total_cores']} cores")
    print(f"RAM: {info['memory']['used_percent']}% of {info['memory']['total_gb']} GB (Available: {info['memory']['available_gb']} GB)")
    print(f"Disk: {info['disk']['used_percent']}% of {info['disk']['total_gb']} GB (Free: {info['disk']['free_gb']} GB)")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the optimized PubMed processing pipeline")
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Directory containing XML files")
    
    # Batch processing arguments
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Number of files to process in each batch")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of worker processes (default: 8)")
    
    # Processing options
    parser.add_argument("--chunk-size", type=int, default=350, help="Target chunk size in tokens")
    parser.add_argument("--model", default="pritamdeka/S-PubMedBERT-MS-MARCO", help="Embedding model name")
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of files to process")
    parser.add_argument("--no-save", action="store_true", help="Don't save intermediate files")
    
    # Output directories
    parser.add_argument("--output-dir", "-o", default="./processed", help="Output directory for processed files")
    parser.add_argument("--embeddings-dir", "-e", default="./processed_with_embeddings", help="Output directory for files with embeddings")
    
    # Weaviate options
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate server URL")
    parser.add_argument("--weaviate-grpc-host", default="localhost", help="Weaviate gRPC host")
    parser.add_argument("--weaviate-grpc-port", type=int, default=50051, help="Weaviate gRPC port")
    parser.add_argument("--collection", default="PubMedArticle", help="Weaviate collection name")
    parser.add_argument("--skip-weaviate", action="store_true", help="Skip loading to Weaviate")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--test", action="store_true", help="Run in test mode with a small number of files")
    parser.add_argument("--test-files", type=int, default=20, help="Number of files to process in test mode (default: 20)")
    
    return parser.parse_args()

def get_file_list(input_dir: str, limit: Optional[int] = None) -> List[Path]:
    """Get a list of XML files from the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    xml_files = list(input_path.glob("**/*.nxml"))
    if not xml_files:
        xml_files = list(input_path.glob("**/*.xml"))
    
    if not xml_files:
        raise ValueError(f"No XML files found in directory: {input_dir}")
    
    logger.info(f"Found {len(xml_files)} XML files in {input_dir}")
    
    if limit and limit > 0:
        xml_files = xml_files[:limit]
        logger.info(f"Limited to {limit} files for processing")
    
    return xml_files

def parse_file(file_path: Path, processor: PubMedProcessor, save_output: bool = False, output_dir: str = "./processed") -> Dict:
    """Parse a single PubMed XML file and return the extracted data."""
    try:
        # Process the file with the save_output option
        document = processor.process_file(file_path, save_output=save_output, output_dir=output_dir)
        return {
            "status": "success",
            "file_path": str(file_path),
            "document": document,
            "chunk_count": len(document.get("chunks", [])) if document else 0
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            "status": "error",
            "file_path": str(file_path),
            "error": str(e)
        }

def generate_embeddings(document: Dict, generator: EmbeddingGenerator, save_output: bool = False, output_dir: str = "./processed_with_embeddings") -> Dict:
    """Generate embeddings for a document's chunks."""
    try:
        # Process the document and add embeddings
        document_with_embeddings = generator.process_document(document, save_output=save_output, output_dir=output_dir)
        return {
            "status": "success",
            "document": document_with_embeddings,
            "embedding_count": len(document_with_embeddings.get("chunks", [])) if document_with_embeddings else 0
        }
    except Exception as e:
        logger.error(f"Error generating embeddings for document {document.get('pmid', 'unknown')}: {str(e)}")
        return {
            "status": "error",
            "document_id": document.get("pmid", "unknown"),
            "error": str(e)
        }

def index_document(document: Dict, weaviate_manager: WeaviateManager) -> Dict:
    """Index a document in Weaviate."""
    try:
        # Index the document in Weaviate
        indexed_count = weaviate_manager.index_document(document)
        return {
            "status": "success",
            "document_id": document.get("pmid", "unknown"),
            "indexed_count": indexed_count
        }
    except Exception as e:
        logger.error(f"Error indexing document {document.get('pmid', 'unknown')}: {str(e)}")
        return {
            "status": "error",
            "document_id": document.get("pmid", "unknown"),
            "error": str(e)
        }

def process_batch(
    batch_files: List[Path],
    processor: PubMedProcessor,
    generator: EmbeddingGenerator,
    weaviate_manager: Optional[WeaviateManager],
    stats: PipelineStats,
    batch_num: int,
    total_batches: int,
    args: argparse.Namespace
) -> Dict:
    """Process a batch of files through the entire pipeline."""
    batch_start_time = time.time()
    batch_stats = {
        "batch_num": batch_num,
        "total_files": len(batch_files),
        "files_processed": 0,
        "files_failed": 0,
        "chunks_generated": 0,
        "chunks_embedded": 0,
        "chunks_indexed": 0
    }
    
    logger.info(f"Starting batch {batch_num}/{total_batches} with {len(batch_files)} files")
    
    # Step 1: Parse files in parallel
    parsed_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(parse_file, file_path, processor, not args.no_save, args.output_dir) for file_path in batch_files]
        
        if not args.no_progress:
            futures_iterator = tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures),
                desc=f"Batch {batch_num}/{total_batches} - Parsing",
                unit="file"
            )
        else:
            futures_iterator = concurrent.futures.as_completed(futures)
            
        for future in futures_iterator:
            result = future.result()
            parsed_results.append(result)
            
            if result["status"] == "success":
                batch_stats["files_processed"] += 1
                batch_stats["chunks_generated"] += result.get("chunk_count", 0)
            else:
                batch_stats["files_failed"] += 1
    
    # Update global stats
    stats.files_processed += batch_stats["files_processed"]
    stats.files_failed += batch_stats["files_failed"]
    stats.chunks_generated += batch_stats["chunks_generated"]
    
    # Step 2: Generate embeddings for successful parses
    embedded_results = []
    successful_documents = [r["document"] for r in parsed_results if r["status"] == "success" and r.get("document")]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_embeddings, doc, generator, not args.no_save, args.embeddings_dir) for doc in successful_documents]
        
        if not args.no_progress:
            futures_iterator = tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures),
                desc=f"Batch {batch_num}/{total_batches} - Embedding",
                unit="doc"
            )
        else:
            futures_iterator = concurrent.futures.as_completed(futures)
            
        for future in futures_iterator:
            result = future.result()
            embedded_results.append(result)
            
            if result["status"] == "success":
                batch_stats["chunks_embedded"] += result.get("embedding_count", 0)
    
    # Update global stats
    stats.chunks_embedded += batch_stats["chunks_embedded"]
    
    # Step 3: Index in Weaviate (if enabled)
    if weaviate_manager and not args.skip_weaviate:
        indexed_results = []
        successful_embeddings = [r["document"] for r in embedded_results if r["status"] == "success" and r.get("document")]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(index_document, doc, weaviate_manager) for doc in successful_embeddings]
            
            if not args.no_progress:
                futures_iterator = tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc=f"Batch {batch_num}/{total_batches} - Indexing",
                    unit="doc"
                )
            else:
                futures_iterator = concurrent.futures.as_completed(futures)
                
            for future in futures_iterator:
                result = future.result()
                indexed_results.append(result)
                
                if result["status"] == "success":
                    batch_stats["chunks_indexed"] += result.get("indexed_count", 0)
        
        # Update global stats
        stats.chunks_indexed += batch_stats["chunks_indexed"]
    
    batch_elapsed_time = time.time() - batch_start_time
    batch_stats["elapsed_time"] = batch_elapsed_time
    batch_stats["files_per_second"] = batch_stats["files_processed"] / batch_elapsed_time if batch_elapsed_time > 0 else 0
    
    logger.info(f"Completed batch {batch_num}/{total_batches} in {batch_elapsed_time:.2f} seconds")
    logger.info(f"  - Files: {batch_stats['files_processed']} processed, {batch_stats['files_failed']} failed")
    logger.info(f"  - Chunks: {batch_stats['chunks_generated']} generated, {batch_stats['chunks_embedded']} embedded, {batch_stats['chunks_indexed']} indexed")
    
    stats.batches_completed += 1
    return batch_stats

def main():
    """Main function to run the optimized pipeline."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Handle test mode
    if args.test:
        args.limit = args.test_files
        logger.info(f"Running in test mode with {args.limit} files")
    
    # Check for required environment variables
    if not args.skip_weaviate and not os.environ.get("WEAVIATE_API_KEY"):
        logger.warning("WEAVIATE_API_KEY environment variable not set. Weaviate indexing may fail.")
    
    # Print system info
    print_system_info()
    
    # Determine number of files to process
    try:
        xml_files = get_file_list(args.input, args.limit)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Configure batch size
    batch_size = min(args.batch_size, len(xml_files))
    num_batches = (len(xml_files) + batch_size - 1) // batch_size
    
    logger.info(f"Starting pipeline with {len(xml_files)} files in {num_batches} batches")
    logger.info(f"Using {args.workers} worker processes")
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    
    # Initialize PubMed processor
    processor_config = {
        "chunk_size": args.chunk_size
    }
    processor = PubMedProcessor(processor_config)
    
    # Initialize embedding generator
    generator_config = {
        "model_name": args.model
    }
    generator = EmbeddingGenerator(generator_config)
    
    # Initialize Weaviate manager (if not skipped)
    weaviate_manager = None
    if not args.skip_weaviate:
        weaviate_config = {
            "weaviate_url": args.weaviate_url,
            "weaviate_grpc_host": args.weaviate_grpc_host,
            "weaviate_grpc_port": args.weaviate_grpc_port,
            "weaviate_api_key": os.environ.get("WEAVIATE_API_KEY"),
            "collection_name": args.collection
        }
        weaviate_manager = WeaviateManager(weaviate_config)
        
        # Check Weaviate connection
        if not weaviate_manager.check_connection():
            logger.error("Failed to connect to Weaviate. Pipeline will continue but indexing will be skipped.")
            args.skip_weaviate = True
            weaviate_manager = None
    
    # Initialize statistics
    stats = PipelineStats()
    
    # Process files in batches
    batch_results = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(xml_files))
        batch_files = xml_files[start_idx:end_idx]
        
        batch_stats = process_batch(
            batch_files=batch_files,
            processor=processor,
            generator=generator,
            weaviate_manager=weaviate_manager,
            stats=stats,
            batch_num=batch_idx + 1,
            total_batches=num_batches,
            args=args
        )
        
        batch_results.append(batch_stats)
        
        # Print intermediate system info after each batch
        print_system_info()
    
    # Print final statistics
    stats.print_summary()
    
    # Check Weaviate final status
    if weaviate_manager and not args.skip_weaviate:
        try:
            collection = weaviate_manager.client.collections.get(args.collection)
            count = collection.query.fetch_objects(limit=1, include_vector=False).total_count
            logger.info(f"Weaviate collection '{args.collection}' now contains {count} objects")
            print(f"\nWeaviate collection '{args.collection}' contains {count} total objects")
        except Exception as e:
            logger.error(f"Failed to get final Weaviate status: {str(e)}")
    
    logger.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())