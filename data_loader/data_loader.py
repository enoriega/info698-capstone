import os
import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset


def load_pubmed_data(
    start_idx: int = 500, num_samples: int = 100
) -> List[Dict[str, Any]]:
    try:
        dataset = load_dataset("MedRAG/pubmed", split="train", streaming=True)

        # Take only the specified samples
        subset = dataset.skip(start_idx).take(num_samples)

        # Convert to list for easy access
        subset_list = list(subset)

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
