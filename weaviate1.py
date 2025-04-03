#!/usr/bin/env python3
"""
Weaviate Manager Script v4 - For PubMed processing pipeline

This script handles storing PubMed chunks with embeddings in Weaviate and
provides search functionality using the v4 client.

Usage:
    # Index data into Weaviate
    python weaviate_manager.py --input /path/to/embeddings [--limit 20]
    
    # Search for data in Weaviate
    python weaviate_manager.py --search "your search query" [--results 5]
    
    # Analyze database contents
    python weaviate_manager.py --analyze [--collection PubMedArticle]
"""

import os
import sys
import json
import logging
import glob
import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from tabulate import tabulate
import time

# Import the v4 Weaviate client
import weaviate
from weaviate.connect import ConnectionParams
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

# Conditionally import for CLI colored output
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weaviate_manager.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("weaviate_manager")

def color_text(text, color=None, style=None):
    """Add color to terminal text if colorama is available."""
    if not HAS_COLORAMA:
        return text
    
    color_code = getattr(Fore, color.upper(), "") if color else ""
    style_code = getattr(Style, style.upper(), "") if style else ""
    reset_code = Style.RESET_ALL
    
    return f"{color_code}{style_code}{text}{reset_code}"

class WeaviateManager:
    def __init__(self, config: Dict = None):
        """Initialize the Weaviate manager with configuration."""
        self.config = {
            "weaviate_url": "http://localhost:8080",
            "weaviate_grpc_host": "localhost",
            "weaviate_grpc_port": 50051,
            "weaviate_api_key": None,
            "collection_name": "PubMedArticle",
            "embedding_model": "pritamdeka/S-PubMedBERT-MS-MARCO"
        }
        if config:
            self.config.update(config)
        
        self.client = None
        self.collection_name = self.config["collection_name"]
        self.connect_to_weaviate()
        
        # Initialize embedding model if needed for search
        self.model = None

    def connect_to_weaviate(self):
        """Connect to Weaviate instance using v4 client."""
        logger.info(f"Connecting to Weaviate at {self.config['weaviate_url']}")
        
        try:
            # Extract http host and port from url
            url_parts = self.config['weaviate_url'].split("://")
            http_secure = url_parts[0] == "https"
            
            if len(url_parts) > 1:
                http_parts = url_parts[1].split(":")
                http_host = http_parts[0]
                http_port = int(http_parts[1]) if len(http_parts) > 1 else 8080
            else:
                logger.error("Invalid Weaviate URL format")
                return
            
            # Create connection params for v4 client
            connection_params = ConnectionParams.from_params(
                http_host=http_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=self.config["weaviate_grpc_host"],
                grpc_port=self.config["weaviate_grpc_port"],
                grpc_secure=False
            )
            
            # Handle authentication if API key provided
            auth = None
            if self.config["weaviate_api_key"]:
                auth = wvc.init.Auth.api_key(self.config["weaviate_api_key"])
            
            # Create client with v4 client
            self.client = weaviate.WeaviateClient(
                connection_params=connection_params,
                auth_client_secret=auth
            )
            
            # Connect to the client
            self.client.connect()
            
            if self.check_connection():
                logger.info("Connected to Weaviate successfully")
            else:
                logger.error("Failed to connect to Weaviate")
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}", exc_info=True)
            self.client = None
    
    def check_connection(self):
        """Check if connection to Weaviate is working."""
        if not self.client:
            return False

        try:
            # Check if Weaviate is ready
            is_ready = self.client.is_ready()
            logger.info(f"Weaviate is ready: {is_ready}")
            return is_ready
        except Exception as e:
            logger.error(f"Error checking Weaviate connection: {str(e)}")
            return False
    
    def create_schema(self):
        """Create collection for PubMed articles in Weaviate."""
        logger.info("Creating collection in Weaviate")
        
        try:
            # Check if collection already exists
            collections = self.client.collections.list_all()
            collection_exists = any(collection.name == self.collection_name for collection in collections)
            
            if collection_exists:
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            # Create collection using v4 client
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=None,  # We'll provide our own vectors
                description="PubMed article chunks with embeddings",
                properties=[
                    wvc.config.Property(
                        name="pmid",
                        data_type=wvc.config.DataType.TEXT,
                        description="PubMed ID"
                    ),
                    wvc.config.Property(
                        name="title",
                        data_type=wvc.config.DataType.TEXT,
                        description="Article title",
                        tokenization=wvc.config.Tokenization.WORD
                    ),
                    wvc.config.Property(
                        name="journal",
                        data_type=wvc.config.DataType.TEXT,
                        description="Journal name"
                    ),
                    wvc.config.Property(
                        name="authors",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="List of author names"
                    ),
                    wvc.config.Property(
                        name="section",
                        data_type=wvc.config.DataType.TEXT,
                        description="Section of the article"
                    ),
                    wvc.config.Property(
                        name="text",
                        data_type=wvc.config.DataType.TEXT,
                        description="The text content",
                        tokenization=wvc.config.Tokenization.WORD
                    ),
                    wvc.config.Property(
                        name="chunk_id",
                        data_type=wvc.config.DataType.TEXT,
                        description="Unique identifier for the chunk"
                    )
                ],
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                )
            )
            
            logger.info(f"Created collection for {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error during collection creation: {str(e)}")
    
    def index_file(self, file_path: str) -> bool:
        """Index a document with embeddings into Weaviate."""
        logger.info(f"Indexing file: {file_path}")
        
        try:
            # Load the document with embeddings
            with open(file_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Extract metadata
            metadata = document.get('metadata', {})
            pmid = metadata.get('pmid', '')
            title = metadata.get('title', '')
            journal = metadata.get('journal', '')
            authors = [author.get('name', '') for author in metadata.get('authors', [])] if 'authors' in metadata else []
            
            # Get chunks with embeddings
            chunks = document.get('chunks', [])
            if not chunks:
                logger.warning(f"No chunks found in {file_path}")
                return False
            
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
                    continue
                
                # Create the object in Weaviate
                try:
                    # Generate a stable UUID based on chunk_id
                    obj_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                    
                    # Store the document with its embedding using v4 client
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
                        vector=embedding  # Pass embedding directly
                    )
                    
                    indexed_count += 1
                except Exception as e:
                    logger.error(f"Error indexing chunk {chunk_id}: {str(e)}")
            
            logger.info(f"Indexed {indexed_count} chunks from {file_path}")
            return indexed_count > 0
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {str(e)}")
            return False
    
    def index_directory(self, directory: str, limit: Optional[int] = None) -> int:
        """Index all files in a directory into Weaviate."""
        logger.info(f"Indexing directory: {directory}")
        
        # Create schema if not exists
        self.create_schema()
        
        # Get all JSON files
        files = sorted(glob.glob(os.path.join(directory, "*.json")))
        if limit:
            files = files[:limit]
        
        logger.info(f"Found {len(files)} files to index")
        
        # Index each file
        indexed_count = 0
        for file_path in files:
            print(f"Indexing file {indexed_count+1}/{len(files)}: {os.path.basename(file_path)}")
            if self.index_file(file_path):
                indexed_count += 1
        
        logger.info(f"Successfully indexed {indexed_count}/{len(files)} files")
        return indexed_count
    
    def load_embedding_model(self):
        """Load the embedding model for search."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config['embedding_model']}")
            self.model = SentenceTransformer(self.config['embedding_model'])
            logger.info("Embedding model loaded")
        return self.model
    
    def search(self, query: str, limit: int = 5, search_type: str = 'vector') -> List[Dict]:
        """Search for articles in Weaviate."""
        logger.info(f"Searching for: {query}")
        
        try:
            # Load embedding model if needed
            self.load_embedding_model()
            
            # Generate embedding for the query
            query_vector = self.model.encode(query).tolist()
            
            # Get the collection
            collection = self.client.collections.get(self.collection_name)
            
            # Perform search based on type using v4 client
            if search_type == 'bm25':
                # BM25 keyword search
                results = collection.query.bm25(
                    query=query,
                    limit=limit
                )
            elif search_type == 'hybrid':
                # Hybrid search (vector + keywords)
                results = collection.query.hybrid(
                    query=query,
                    limit=limit
                )
            else:
                # Vector similarity search (default)
                results = collection.query.near_vector(
                    vector=query_vector,
                    limit=limit
                )
            
            # Process results from v4 client
            search_results = []
            
            for obj in results.objects:
                search_results.append({
                    "pmid": obj.properties.get("pmid", ""),
                    "title": obj.properties.get("title", ""),
                    "journal": obj.properties.get("journal", ""),
                    "authors": obj.properties.get("authors", []),
                    "section": obj.properties.get("section", ""),
                    "text": obj.properties.get("text", ""),
                    "chunk_id": obj.properties.get("chunk_id", ""),
                    "score": getattr(obj.metadata, "distance", None)
                })
            
            logger.info(f"Found {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}", exc_info=True)
            return []
    
    def analyze_database(self):
        """Analyze the Weaviate database contents."""
        logger.info("Analyzing database")
        
        try:
            # Get database info
            print("\n=== WEAVIATE DATABASE INFO ===")
            meta = self.client.get_meta()
            print(f"Version: {meta.get('version', 'N/A')}")
            
            # List collections
            print("\n=== COLLECTIONS ===")
            collections = self.client.collections.list_all()
            print(f"Total collections: {len(collections)}")
            
            collection_info = []
            for collection in collections:
                try:
                    # Get object count
                    count_result = collection.aggregate.over_all()
                    total_count = count_result.total_count
                except Exception as e:
                    total_count = f"Error: {str(e)}"
                
                collection_info.append({
                    "Name": collection.name,
                    "Description": collection.description or "No description",
                    "Object Count": total_count,
                    "Properties": len(collection.properties)
                })
            
            # Display as table
            if collection_info:
                df = pd.DataFrame(collection_info)
                print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
            
            # Analyze target collection
            if any(collection.name == self.collection_name for collection in collections):
                self.analyze_collection(self.collection_name)
            else:
                print(f"\nCollection {self.collection_name} not found")
        
        except Exception as e:
            logger.error(f"Error analyzing database: {str(e)}")
    
    def analyze_collection(self, collection_name: str):
        """Analyze a specific collection in detail."""
        logger.info(f"Analyzing collection: {collection_name}")
        
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            print(f"\n=== COLLECTION: {collection_name} ===")
            print(f"Description: {collection.description}")
            
            # Get schema
            print("\n=== SCHEMA ===")
            schema_info = []
            for prop in collection.properties:
                schema_info.append({
                    "Name": prop.name,
                    "Data Type": prop.data_type,
                    "Description": prop.description or "No description"
                })
            
            if schema_info:
                df_schema = pd.DataFrame(schema_info)
                print(tabulate(df_schema, headers='keys', tablefmt='grid', showindex=False))
            
            # Get object count
            try:
                count_result = collection.aggregate.over_all()
                total_count = count_result.total_count
                print(f"\nTotal objects: {total_count:,}")
            except Exception as e:
                print(f"\nError getting object count: {str(e)}")
                total_count = 0
            
            # Check for PMID distribution if applicable
            if 'pmid' in [prop.name for prop in collection.properties] and total_count > 0:
                print("\n=== PMID DISTRIBUTION ===")
                try:
                    # Group by PMID
                    pmid_groups = collection.aggregate.group_by(
                        property_name="pmid",
                        limit=50
                    )
                    
                    if hasattr(pmid_groups, 'groups'):
                        # Calculate statistics
                        pmid_counts = [group.count for group in pmid_groups.groups]
                        if pmid_counts:
                            avg_chunks = sum(pmid_counts) / len(pmid_counts)
                            max_chunks = max(pmid_counts)
                            min_chunks = min(pmid_counts)
                            
                            print(f"Documents analyzed: {len(pmid_counts)}")
                            print(f"Average chunks per document: {avg_chunks:.2f}")
                            print(f"Maximum chunks per document: {max_chunks}")
                            print(f"Minimum chunks per document: {min_chunks}")
                            
                            # Show top documents
                            print("\nTop 10 documents by chunk count:")
                            top_docs = []
                            for i, group in enumerate(sorted(pmid_groups.groups, key=lambda x: x.count, reverse=True)[:10]):
                                top_docs.append({
                                    "PMID": group.value,
                                    "Chunk Count": group.count
                                })
                            
                            if top_docs:
                                top_docs_df = pd.DataFrame(top_docs)
                                print(tabulate(top_docs_df, headers='keys', tablefmt='grid', showindex=False))
                except Exception as e:
                    print(f"Error analyzing PMID distribution: {str(e)}")
            
            # Get sample objects if available
            if total_count > 0:
                print("\n=== SAMPLE OBJECTS ===")
                try:
                    sample_count = min(3, total_count)
                    for i, obj in enumerate(collection.iterator()):
                        if i >= sample_count:
                            break
                        
                        print(f"\nObject {i+1} (ID: {obj.uuid}):")
                        for key, value in obj.properties.items():
                            if isinstance(value, str) and len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                except Exception as e:
                    print(f"Error retrieving sample objects: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error analyzing collection {collection_name}: {str(e)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Weaviate Manager for PubMed data")
    
    # Create mutually exclusive group for operation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Directory containing JSON files with embeddings to index")
    group.add_argument("--search", "-s", help="Search query to execute")
    group.add_argument("--analyze", "-a", action="store_true", help="Analyze database contents")
    
    # Other parameters
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of files to process")
    parser.add_argument("--results", "-r", type=int, default=5, help="Number of search results to return")
    parser.add_argument("--search-type", "-t", choices=["vector", "bm25", "hybrid"], default="hybrid", 
                        help="Type of search to perform")
    parser.add_argument("--url", default="http://localhost:8080", help="Weaviate server URL")
    parser.add_argument("--grpc-host", default="localhost", help="Weaviate gRPC host")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Weaviate gRPC port")
    parser.add_argument("--api-key", help="Weaviate API key")
    parser.add_argument("--collection", default="PubMedArticle", help="Weaviate collection name")
    parser.add_argument("--model", default="pritamdeka/S-PubMedBERT-MS-MARCO", 
                        help="Embedding model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()


def display_search_results(results, limit=5):
    """Format and display search results in CLI."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results[:limit], 1):
        title = color_text(result['title'], 'green', 'bright') if HAS_COLORAMA else result['title']
        pmid = color_text(f"PMID: {result['pmid']}", 'blue') if HAS_COLORAMA else f"PMID: {result['pmid']}"
        journal = color_text(result['journal'], 'yellow') if HAS_COLORAMA else result['journal']
        
        print(f"{'='*30} Result {i} {'='*30}")
        print(f"{title}")
        print(f"{pmid} | {journal}")
        print(f"Section: {result['section']}")
        
        # Format authors if available
        if result.get('authors'):
            authors = ', '.join(result['authors'][:3])
            if len(result['authors']) > 3:
                authors += f" and {len(result['authors'])-3} more"
            print(f"Authors: {authors}")
        
        # Print score with color based on value
        score = result.get('score')
        if score is not None:
            if HAS_COLORAMA:
                if score > 0.9:
                    score_text = color_text(f"Score: {score:.4f}", 'green', 'bright')
                elif score > 0.7:
                    score_text = color_text(f"Score: {score:.4f}", 'green')
                elif score > 0.5:
                    score_text = color_text(f"Score: {score:.4f}", 'yellow')
                else:
                    score_text = color_text(f"Score: {score:.4f}", 'red')
            else:
                score_text = f"Score: {score:.4f}"
            print(score_text)
        
        # Print text with context highlighting
        text = result['text']
        if len(text) > 500:
            text = text[:500] + "..."
        print(f"\n{text}\n")


def main():
    """Main function to run the Weaviate manager."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create config from arguments
    config = {
        "weaviate_url": args.url,
        "weaviate_grpc_host": args.grpc_host,
        "weaviate_grpc_port": args.grpc_port,
        "weaviate_api_key": args.api_key,
        "collection_name": args.collection,
        "embedding_model": args.model
    }
    
    # Create Weaviate manager
    manager = WeaviateManager(config)
    
    # Check if we're in index mode
    if args.input:
        # Index data into Weaviate
        indexed_count = manager.index_directory(args.input, args.limit)
        print(f"Successfully indexed {indexed_count} files into Weaviate collection '{args.collection}'")
    
    # Check if we're in search mode
    elif args.search:
        # Load the embedding model
        manager.load_embedding_model()
        
        # Perform search
        print(f"Searching for: {args.search}")
        print(f"Search type: {args.search_type}")
        
        results = manager.search(args.search, limit=args.results, search_type=args.search_type)
        
        # Display results
        display_search_results(results, args.results)
    
    # Check if we're in analyze mode
    elif args.analyze:
        # Analyze database contents
        manager.analyze_database()


if __name__ == "__main__":
    main()