import os
import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
import logging

import streamlit as st
import json
import os
import logging
from dotenv import load_dotenv
# Configure logging - simplified
logging.basicConfig(filename="dataloader.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)



from supabase import create_client, Client
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


# Creating an object
logger = logging.getLogger()


def load_pubmed_data(
    start_idx, num_samples
) -> List[Dict[str, Any]]:
    try:
        dataset = load_dataset("MedRAG/pubmed", split="train", streaming=True)

        # Take only the specified samples
        subset = dataset.skip(start_idx).take(num_samples)

        # Convert to list for easy access
        subset_list = list(subset)
        logger.debug("Size of subset list ", len(subset_list))
        # Format documents with consistent structure
        documents = [
            {
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "contents": item["contents"],
                "PMID": item["PMID"],
            }
            for item in subset_list
        ]

        return documents

    except Exception as e:
        print(f"Error loading PubMed data: {str(e)}")
        return []
    
def process_large_dataset(supabase_instance, start_index=10000, total_rows=10000, batch_size=1000):
    """
    Process large number of rows in batches using create_new_vector_store

    Args:
        supabase_instance: Supabase client instance
        start_index (int): Starting index in the dataset (default 0)
        total_rows (int): Total number of rows to process (default 10000)
        batch_size (int): Number of samples to process in each batch (default 100)
    """
    try:
        # Calculate number of iterations needed
        num_iterations = (total_rows + batch_size - 1) // batch_size

        for i in range(num_iterations):
            current_start_idx = start_index + (i * batch_size)
            current_batch_size = min(batch_size, total_rows - (i * batch_size))

            print(f"\nProcessing batch {i+1}/{num_iterations}")
            print(f"Start index: {current_start_idx}, Batch size: {current_batch_size}")

            # Call create_new_vector_store for this batch
            create_new_vector_store(
                supabase_instance,
                start_idx=current_start_idx,
                num_samples=current_batch_size
            )

            print(f"Completed batch {i+1}/{num_iterations}")

        print(f"\nCompleted processing all {total_rows} rows starting from index {start_index}")

    except Exception as e:
        print(f"Error in process_large_dataset: {str(e)}")
        
def create_new_vector_store(
    supabase_instance, start_idx, num_samples
):
    try:
        documents = load_pubmed_data(start_idx, num_samples)
        print("Documents lenght: ", len(documents))
        # Add a check out of all documents which have PMID value.
        for doc in documents:
            if not doc.get("PMID"):
                print(f"Document with ID {doc['id']} does not have a PMID value.")
                continue

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [doc["content"] for doc in documents]
        embeddings = model.encode(texts).tolist()
        data = []
        for doc, embedding in zip(documents, embeddings):
            record = {
                "title": doc["title"],
                "pmid": str(doc["PMID"]),
                "content": doc["content"],
                "metadata": {"original_id": doc["id"], "contents": doc["contents"]},
                "embedding": embedding,
            }
            data.append(record)

        # Insert data into Supabase table
        result = supabase_instance.table("pubmed_documents").upsert(data).execute()
        print(f"Inserted document with title: {['title']}")
        
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        

# Load environment variables
load_dotenv()
# LLM_APIKEY = os.getenv("LLM_APIKEY")

def main():

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    logger.debug("Supabase create client")
    supabase_client = create_client(supabase_url, supabase_key)
    # Change the start index and total rows as per your requirement
    process_large_dataset(supabase_client, start_index=10001, total_rows=10000, batch_size=1000)
    
        
if __name__ == "__main__":
    main()