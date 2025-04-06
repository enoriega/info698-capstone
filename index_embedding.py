#!/usr/bin/env python3
"""
Optimized Script to index document embeddings to Weaviate
This script is designed to work as part of a batch processing pipeline
It can process documents directly in memory or from files
"""

import os
import json
import glob
import logging
import argparse
import uuid
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil

try:
    from tqdm import tqdm
    import weaviate
    from weaviate.classes.config import Property, DataType, Configure, Tokenization, VectorDistances
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install required packages manually with:")
    print("pip install weaviate-client==4.4.4 typing-extensions tqdm psutil")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weaviate_indexer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("weaviate_indexer")

# Thread-safe counters
class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()
        
    def increment(self, delta=1):
        with self._lock:
            old = self._value
            self._value += delta
            return old
            
    def value(self):
        with self._lock:
            return self._value
            
    def reset(self):
        with self._lock:
            self._value = 0

class WeaviateManager:
    def __init__(self, url="http://149.165.151.58:8080", grpc_host="149.165.151.58", grpc_port=50051,
                 api_key="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE="):
        """Initialize connection to Weaviate."""
        logger.info(f"Connecting to Weaviate at {url}")
        
        # Extract http host and port from url
        http_parts = url.split("://")[1].split(":")
        http_host = http_parts[0]
        http_port = int(http_parts[1]) if len(http_parts) > 1 else 8080
        http_secure = url.startswith("https")
        
        # Setup counters for statistics
        self.total_documents_processed = AtomicCounter()
        self.total_chunks_indexed = AtomicCounter()
        self.total_chunks_failed = AtomicCounter()
        self.start_time = time.time()
        
        try:
            # Import the required classes
            from weaviate.connect import ConnectionParams
            import weaviate.classes.init as wvc_init
            
            # Create connection params
            connection_params = ConnectionParams.from_params(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=False
            )
            
            # Create client using WeaviateClient
            self.client = weaviate.WeaviateClient(
                connection_params=connection_params,
                auth_client_secret=wvc_init.Auth.api_key(api_key)
            )
            
            # Connect the client explicitly
            self.client.connect()
            
            self.check_connection()
            logger.info("Connected to Weaviate successfully")
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            self.client = None
            
        # Set the collection name for medical articles
        self.collection_name = "PubMedArticle27"

    def check_connection(self):
        """Check if connection to Weaviate is working."""
        if not self.client:
            return False
            
        try:
            # Make sure client is connected
            if not self.client._connection.is_connected:
                self.client.connect()
                
            # Check if Weaviate is ready
            is_ready = self.client.is_ready()
            logger.info(f"Weaviate is ready: {is_ready}")
            return is_ready
        except Exception as e:
            logger.error(f"Error checking Weaviate connection: {str(e)}")
            return False

    def create_schema(self):
        """Create collection for PubMed articles in Weaviate."""
        logger.info("Checking/creating collection in Weaviate")
        
        try:
            # Make sure client is connected
            if not self.client._connection.is_connected:
                self.client.connect()
                
            collections = self.client.collections.list_all()
            collection_exists = any(collection.name == self.collection_name for collection in collections)
                
            if collection_exists:
                logger.info(f"Collection {self.collection_name} already exists")
                return
                
            # Define the collection configuration with properties
            collection = self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=None,  # We'll provide our own vectors
                description="PubMed article chunks with embeddings",
                properties=[
                    Property(
                        name="pmid",
                        data_type=DataType.TEXT,
                        description="PubMed ID"
                    ),
                    Property(
                        name="title",
                        data_type=DataType.TEXT,
                        description="Article title",
                        tokenization=Tokenization.WORD
                    ),
                    Property(
                        name="journal",
                        data_type=DataType.TEXT,
                        description="Journal name"
                    ),
                    Property(
                        name="authors",
                        data_type=DataType.TEXT_ARRAY,
                        description="List of author names"
                    ),
                    Property(
                        name="section",
                        data_type=DataType.TEXT,
                        description="Section of the article"
                    ),
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="The text content",
                        tokenization=Tokenization.WORD
                    ),
                    Property(
                        name="chunk_id",
                        data_type=DataType.TEXT,
                        description="Unique identifier for the chunk"
                    )
                ],
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            )
            
            logger.info(f"Created collection {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error during collection creation: {str(e)}")
    
    def index_document(self, document):
        """Index a document with embeddings into Weaviate."""
        try:
            # Make sure client is connected
            if not self.client._connection.is_connected:
                self.client.connect()
                
            # Extract metadata
            metadata = document.get('metadata', {})
            pmid = metadata.get('pmid', '')
            title = metadata.get('title', '')
            journal = metadata.get('journal', '')
            authors = [author.get('name', '') for author in metadata.get('authors', [])] if 'authors' in metadata else []
            
            # Get chunks with embeddings
            chunks = document.get('chunks', [])
            if not chunks:
                logger.warning(f"No chunks found in document {pmid}")
                return 0
                
            # Get the collection
            collection = self.client.collections.get(self.collection_name)
            
            # Index each chunk
            indexed_count = 0
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', '')
                section = chunk.get('section', '')
                text = chunk.get('text', '')
                embedding = chunk.get('embedding', None)
                
                # Skip chunks without embeddings
                if not embedding:
                    logger.warning(f"No embedding found for chunk {chunk_id}")
                    self.total_chunks_failed.increment()
                    continue
                    
                # Create the object in Weaviate
                try:
                    # Generate a stable UUID based on chunk_id
                    obj_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
                    
                    # Pass the embedding vector directly to vector parameter
                    collection.data.insert(
                        properties={
                            "pmid": pmid,
                            "title": title,
                            "journal": journal,
                            "authors": authors,
                            "section": section,
                            "text": text,
                            "chunk_id": chunk_id
                        },
                        uuid=obj_id,
                        vector=embedding  # Store embedding directly
                    )
                    
                    indexed_count += 1
                    self.total_chunks_indexed.increment()
                except Exception as e:
                    logger.error(f"Error indexing chunk {chunk_id}: {str(e)}")
                    self.total_chunks_failed.increment()
                    
            self.total_documents_processed.increment()
            logger.debug(f"Indexed {indexed_count} chunks from document {pmid}")
            return indexed_count
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return 0
    
    def index_file(self, file_path):
        """Index a document file with embeddings into Weaviate."""
        logger.debug(f"Indexing file: {file_path}")
        
        try:
            # Load the document with embeddings
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
                
            return self.index_document(document)
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
            return 0
    
    def process_batch(self, batch, from_files=True):
        """Process a batch of documents or files."""
        logger.info(f"Processing batch of {len(batch)} {'files' if from_files else 'documents'}")
        
        # Create a thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            if from_files:
                # Process files in batch
                results = list(tqdm(
                    executor.map(self.index_file, batch),
                    total=len(batch),
                    desc="Indexing files"
                ))
            else:
                # Process documents in batch
                results = list(tqdm(
                    executor.map(self.index_document, batch),
                    total=len(batch),
                    desc="Indexing documents"
                ))
                
        # Calculate batch statistics
        batch_chunks = sum(results)
        logger.info(f"Batch complete: {len(batch)} {'files' if from_files else 'documents'} processed, {batch_chunks} chunks indexed")
        
        # Report memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        return batch_chunks
    
    def index_directory(self, directory, batch_size=100):
        """Index all files in a directory into Weaviate with batch processing."""
        logger.info(f"Indexing directory: {directory} with batch size {batch_size}")
        
        # Create schema if not exists
        self.create_schema()
        
        # Get all JSON files
        files = sorted(glob.glob(os.path.join(directory, "*.json")))
        total_files = len(files)
        logger.info(f"Found {total_files} files to index")
        
        # Process files in batches
        batch_count = 0
        for i in range(0, total_files, batch_size):
            batch_count += 1
            batch = files[i:i+batch_size]
            logger.info(f"Processing batch {batch_count} ({len(batch)} files)")
            
            # Process this batch
            self.process_batch(batch, from_files=True)
            
            # Log progress
            processed = min(i + batch_size, total_files)
            logger.info(f"Progress: {processed}/{total_files} files ({processed/total_files*100:.1f}%)")
            logger.info(f"Total chunks indexed so far: {self.total_chunks_indexed.value()}")
            logger.info(f"Total chunks failed so far: {self.total_chunks_failed.value()}")
            
        # Log final statistics
        self.log_statistics()
        return self.total_chunks_indexed.value()
    
    def process_documents(self, documents, batch_size=100):
        """Process a list of documents directly (without files) in batches."""
        logger.info(f"Processing {len(documents)} documents with batch size {batch_size}")
        
        # Create schema if not exists
        self.create_schema()
        
        # Process documents in batches
        total_docs = len(documents)
        batch_count = 0
        for i in range(0, total_docs, batch_size):
            batch_count += 1
            batch = documents[i:i+batch_size]
            logger.info(f"Processing batch {batch_count} ({len(batch)} documents)")
            
            # Process this batch
            self.process_batch(batch, from_files=False)
            
            # Log progress
            processed = min(i + batch_size, total_docs)
            logger.info(f"Progress: {processed}/{total_docs} documents ({processed/total_docs*100:.1f}%)")
            logger.info(f"Total chunks indexed so far: {self.total_chunks_indexed.value()}")
            logger.info(f"Total chunks failed so far: {self.total_chunks_failed.value()}")
            
        # Log final statistics
        self.log_statistics()
        return self.total_chunks_indexed.value()
        
    def get_collection_stats(self):
        """Get statistics about the collection in Weaviate."""
        try:
            if not self.client._connection.is_connected:
                self.client.connect()
                
            collection = self.client.collections.get(self.collection_name)
            stats = {
                "collection_name": self.collection_name,
                "object_count": collection.aggregate.over_all(total_count=True).total_count
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"collection_name": self.collection_name, "object_count": "unknown"}
            
    def log_statistics(self):
        """Log detailed statistics about the indexing process."""
        elapsed_time = time.time() - self.start_time
        collection_stats = self.get_collection_stats()
        
        logger.info("=" * 50)
        logger.info("WEAVIATE INDEXING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total documents processed: {self.total_documents_processed.value()}")
        logger.info(f"Total chunks indexed: {self.total_chunks_indexed.value()}")
        logger.info(f"Total chunks failed: {self.total_chunks_failed.value()}")
        logger.info(f"Average chunks per document: {self.total_chunks_indexed.value() / max(1, self.total_documents_processed.value()):.2f}")
        logger.info(f"Total time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Indexing rate: {self.total_chunks_indexed.value() / max(1, elapsed_time):.2f} chunks/second")
        logger.info(f"Collection stats: {collection_stats}")
        logger.info("=" * 50)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Index document embeddings to Weaviate")
    parser.add_argument("--weaviate_url", type=str, default="http://149.165.151.58:8080", 
                        help="Weaviate HTTP URL")
    parser.add_argument("--grpc_host", type=str, default="149.165.151.58", 
                        help="Weaviate gRPC host")
    parser.add_argument("--grpc_port", type=int, default=50051, 
                        help="Weaviate gRPC port")
    parser.add_argument("--api_key", type=str, 
                        default="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE=", 
                        help="Weaviate API key")
    parser.add_argument("--embeddings_dir", type=str, required=True, 
                        help="Directory containing embedding JSON files")
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Number of files to process in each batch")
    
    args = parser.parse_args()
    
    # Initialize Weaviate Manager
    weaviate_manager = WeaviateManager(
        url=args.weaviate_url,
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        api_key=args.api_key
    )
    
    logger.info("Weaviate Manager initialized")
    if not weaviate_manager.check_connection():
        logger.error("Failed to connect to Weaviate. Check connection settings and try again.")
        sys.exit(1)
    
    # Index files with embeddings to Weaviate
    start_time = time.time()
    chunks_indexed = weaviate_manager.index_directory(
        directory=args.embeddings_dir,
        batch_size=args.batch_size
    )
    
    # Print final summary
    elapsed = time.time() - start_time
    logger.info(f"Indexing complete: {chunks_indexed} chunks indexed in {elapsed:.2f} seconds")
    collection_stats = weaviate_manager.get_collection_stats()
    logger.info(f"Collection {collection_stats['collection_name']} now contains {collection_stats['object_count']} objects")

if __name__ == "__main__":
    main()