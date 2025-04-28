import os
from pubmed_parser import PubMedProcessor  # Import your nXML parser class
from embedding_generator import EmbeddingGenerator  # Import your embedding generator class
from insert_row import WeaviateManager  # Import the WeaviateManager class


class PipelineManager:
    def __init__(self, raw_data_dir, chunked_data_dir, embedded_data_dir, weaviate_url, weaviate_api_key, file_limit=None):
        """
        Initialize the pipeline with directories, Weaviate configuration, and file limit.
        """
        self.raw_data_dir = raw_data_dir  # Directory containing raw .nxml files
        self.chunked_data_dir = chunked_data_dir  # Directory to store chunked JSON files
        self.embedded_data_dir = embedded_data_dir  # Directory to store JSON files with embeddings
        self.file_limit = file_limit  # Limit on the number of files to process

        # Initialize the WeaviateManager
        self.weaviate_manager = WeaviateManager(
            url=weaviate_url,
            api_key=weaviate_api_key,
            class_name="test_db",
            schema={
                "class": "test_db",
                "description": "Database for PubMed articles with metadata and text chunks",
                "vectorizer": "none",  # We'll provide vectors externally
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "efConstruction": 128,
                    "maxConnections": 32
                },
                "properties": [
                    {"name": "pmid", "dataType": ["text"], "description": "PubMed ID"},
                    {"name": "pmc", "dataType": ["text"], "description": "PubMed Central ID"},
                    {"name": "doi", "dataType": ["text"], "description": "Digital Object Identifier"},
                    {"name": "title", "dataType": ["text"], "description": "Article title"},
                    {"name": "journal", "dataType": ["text"], "description": "Journal name"},
                    {"name": "issn", "dataType": ["text"], "description": "International Standard Serial Number"},
                    {"name": "volume", "dataType": ["text"], "description": "Journal volume"},
                    {"name": "issue", "dataType": ["text"], "description": "Journal issue"},
                    {"name": "publication_date", "dataType": ["text"], "description": "Publication date"},
                    {"name": "authors", "dataType": ["text[]"], "description": "List of author names"},
                    {"name": "keywords", "dataType": ["text[]"], "description": "List of keywords"},
                    {"name": "text", "dataType": ["text"], "description": "Article text content"},
                    {"name": "chunk_id", "dataType": ["text"], "description": "Unique identifier for text chunk"}
                ]
            }
        )

    def run_pipeline(self):
        """
        Run the full pipeline: parse, generate embeddings, and upload to Weaviate.
        """
        print("Starting the pipeline...")

        # Step 1: Parse .nxml files and generate chunked JSON files
        print("Step 1: Parsing .nxml files...")
        self.parse_nxml_files()

        # Step 2: Generate embeddings for the chunked JSON files
        print("Step 2: Generating embeddings...")
        self.generate_embeddings()

        # Step 3: Upload the embedded JSON files to Weaviate
        print("Step 3: Uploading data to Weaviate...")
        self.upload_to_weaviate()

        print("Pipeline completed successfully!")

    def parse_nxml_files(self):
        """
        Parse .nxml files and generate chunked JSON files.
        """
        # Configure the PubMedProcessor
        parser_config = {
            "output_dir": self.chunked_data_dir,
            "chunk_size": 350,
            "min_chunk_size": 100,
            "max_chunk_size": 500,
            "no_save": False
        }
        parser = PubMedProcessor(config=parser_config)

        # Process all .nxml files in the raw data directory with a file limit
        parser.process_directory(self.raw_data_dir, file_limit=self.file_limit)

    def generate_embeddings(self):
        """
        Generate embeddings for the chunked JSON files.
        """
        # Configure the EmbeddingGenerator
        embedding_config = {
            "model_name": "pritamdeka/S-PubMedBERT-MS-MARCO",
            "batch_size": 32,
            "output_dir": self.embedded_data_dir,
            "no_save": False
        }
        generator = EmbeddingGenerator(config=embedding_config)

        # Process all chunked JSON files in the chunked data directory with a file limit
        generator.process_directory(self.chunked_data_dir, self.embedded_data_dir, limit=self.file_limit)

    def upload_to_weaviate(self):
        """
        Upload the embedded JSON files to the Weaviate database.
        """
        # Set up the schema in Weaviate
        self.weaviate_manager.setup_schema()

        # Insert data from the embedded data directory
        self.weaviate_manager.insert_data_from_folder(self.embedded_data_dir)


# Main execution
if __name__ == "__main__":
    # Define directories
    raw_data_dir = r"data/extracted/xml"  # Directory with .nxml files
    chunked_data_dir = r"data/processed"  # Directory for chunked JSON files
    embedded_data_dir = r"data/embedded"  # Directory for JSON files with embeddings

    # Weaviate configuration
    weaviate_url = "http://149.165.151.58:8080"
    weaviate_api_key = "ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE="

    # File limit (set to None for no limit)
    file_limit = None  # Change this value to limit the number of files processed
    

    # Initialize and run the pipeline
    pipeline = PipelineManager(
        raw_data_dir=raw_data_dir,
        chunked_data_dir=chunked_data_dir,
        embedded_data_dir=embedded_data_dir,
        weaviate_url=weaviate_url,
        weaviate_api_key=weaviate_api_key,
        file_limit=file_limit
    )
    pipeline.run_pipeline()