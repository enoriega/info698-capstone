import weaviate
from weaviate.auth import AuthApiKey
import json
from datetime import datetime

# Initialize Weaviate client with API key
client = weaviate.Client(
    url="http://149.165.151.58:8080",
    auth_client_secret=AuthApiKey(api_key="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE=")
)

# Define the schema
schema = {
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
        {
            "name": "pmid",
            "dataType": ["text"],
            "description": "PubMed ID"
        },
        {
            "name": "pmc",
            "dataType": ["text"],
            "description": "PubMed Central ID"
        },
        {
            "name": "doi",
            "dataType": ["text"],
            "description": "Digital Object Identifier"
        },
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Article title"
        },
        {
            "name": "journal",
            "dataType": ["text"],
            "description": "Journal name"
        },
        {
            "name": "issn",
            "dataType": ["text"],
            "description": "International Standard Serial Number"
        },
        {
            "name": "volume",
            "dataType": ["text"],
            "description": "Journal volume"
        },
        {
            "name": "issue",
            "dataType": ["text"],
            "description": "Journal issue"
        },
        {
            "name": "publication_date",
            "dataType": ["text"],
            "description": "Publication date"
        },
        {
            "name": "authors",
            "dataType": ["text[]"],
            "description": "List of author names"
        },
        {
            "name": "keywords",
            "dataType": ["text[]"],
            "description": "List of keywords"
        },
        {
            "name": "text",
            "dataType": ["text"],
            "description": "Article text content"
        },
        {
            "name": "chunk_id",
            "dataType": ["text"],
            "description": "Unique identifier for text chunk"
        }
    ]
}

# First, check if the class exists and delete it if it does
try:
    if client.schema.exists("test_db"):
        client.schema.delete_class("test_db")
    print("Successfully checked/deleted existing schema")
except Exception as e:
    print(f"Error checking/deleting schema: {str(e)}")

# Create the schema
try:
    client.schema.create_class(schema)
    print("Successfully created new schema")
except Exception as e:
    print(f"Error creating schema: {str(e)}")

# Load the embeddings file
with open('data_preprocess/embed_files.json', 'r') as f:
    embed_data = json.load(f)

# Process the first chunk as a sample
sample_chunk = embed_data['chunks'][0]

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
    "text": sample_chunk["text"],
    "chunk_id": sample_chunk["chunk_id"]
}

# Insert the data with the pre-generated embedding
try:
    client.data_object.create(
        class_name="test_db",
        data_object=data_object,
        vector=sample_chunk["embedding"]  # Using the pre-generated embedding
    )
    print("Successfully inserted the data with the pre-generated embedding")

    # Verify the insertion
    result = client.query.get(
        "test_db", ["pmid", "chunk_id", "text"]
    ).with_limit(1).do()

    print("\nVerification of inserted data:")
    print(json.dumps(result, indent=2))

    # Test vector search
    vector_query = (
        client.query
        .get("test_db", ["pmid", "chunk_id", "text"])
        .with_near_vector({
            "vector": sample_chunk["embedding"]
        })
        .with_limit(1)
        .do()
    )
    print("\nVector search test result:")
    print(json.dumps(vector_query, indent=2))

except Exception as e:
    print(f"Error occurred during data insertion or query: {str(e)}")