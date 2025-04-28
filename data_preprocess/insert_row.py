import weaviate
from weaviate.auth import AuthApiKey
import json
import os


class WeaviateManager:
    def __init__(self, url, api_key, class_name, schema):
        """
        Initialize the Weaviate client and set up the schema.
        """
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=AuthApiKey(api_key=api_key)
        )
        self.class_name = class_name
        self.schema = schema

    def setup_schema(self):
        """
        Check if the schema exists, delete it if it does, and create a new schema.
        """
        try:
            if self.client.schema.exists(self.class_name):
                self.client.schema.delete_class(self.class_name)
                print(f"Deleted existing schema: {self.class_name}")
            self.client.schema.create_class(self.schema)
            print(f"Successfully created schema: {self.class_name}")
        except Exception as e:
            print(f"Error setting up schema: {str(e)}")

    def insert_data_from_folder(self, folder_path):
        """
        Iterate through all JSON files in the folder and insert data into Weaviate.
        """
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Skip non-JSON files
            if not file_name.endswith(".json"):
                print(f"Skipping non-JSON file: {file_name}")
                continue

            try:
                # Load the embedding file
                with open(file_path, 'r',encoding="utf-8") as f:
                    embed_data = json.load(f)

                # Process each chunk in the file
                for chunk in embed_data['chunks']:
                    # Prepare the data object
                    data_object = {
                        "pmid": embed_data["metadata"]["pmid"],
                        "pmc": embed_data["metadata"]["pmc"],
                        "doi": embed_data["metadata"]["doi"],
                        "title": embed_data["metadata"]["title"],
                        "journal": embed_data["metadata"]["journal"],
                        "issn": embed_data["metadata"]["issn"],
                        "volume": embed_data["metadata"]["volume"],
                        "issue": embed_data["metadata"]["issue"],
                        "publication_date": embed_data["metadata"]["publication_date"],
                        "authors": [author["name"] for author in embed_data["metadata"]["authors"]],
                        "keywords": embed_data["metadata"]["keywords"],
                        "text": chunk["text"],
                        "chunk_id": chunk["chunk_id"]
                    }

                    # Insert the data with the pre-generated embedding
                    self.client.data_object.create(
                        class_name=self.class_name,
                        data_object=data_object,
                        vector=chunk["embedding"]  # Using the pre-generated embedding
                    )
                print(f"Successfully inserted data from file: {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")

    def verify_insertion(self, limit=5):
        """
        Verify the data insertion by querying a sample of the data.
        """
        try:
            result = self.client.query.get(
                self.class_name, ["pmid", "chunk_id", "text"]
            ).with_limit(limit).do()

            print("\nVerification of inserted data:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error verifying data: {str(e)}")


# Define the schema
schema = {
    "class": "pubmed_db",
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

# Main execution
if __name__ == "__main__":
    # Initialize the WeaviateManager
    weaviate_manager = WeaviateManager(
        url="http://149.165.151.58:8080",
        api_key="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE=",
        class_name="pubmed_db",
        schema=schema
    )

    # Set up the schema
    weaviate_manager.setup_schema()

    # Path to the folder containing embedding files
    embedding_folder = r"data/embedded"

    # Insert data from the folder
    weaviate_manager.insert_data_from_folder(embedding_folder)

    # Verify the insertion
    weaviate_manager.verify_insertion(limit=5)