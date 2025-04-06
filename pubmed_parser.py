#!/usr/bin/env python3
"""
PubMed Parser Script - First step in PubMed processing pipeline

This script handles parsing PubMed XML files, extracting metadata and text,
and chunking the text into manageable pieces.

Usage:
    python 1_pubmed_parser.py --input /path/to/xmls --batch-size 100 --workers 8 [--limit 20] [--no-save]
"""

import os
import sys
import json
import logging
import re
import glob
import argparse
import unicodedata
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm
from lxml import etree
from multiprocessing import Pool, Manager, Value, Lock
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubmed_parser.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pubmed_parser")

# Global counters with thread safety
class ProcessingStats:
    def __init__(self):
        self.processed_count = Value('i', 0)
        self.failed_count = Value('i', 0)
        self.total_chunks = Value('i', 0)
        self.lock = Lock()
        
    def update(self, processed=0, failed=0, chunks=0):
        with self.lock:
            self.processed_count.value += processed
            self.failed_count.value += failed
            self.total_chunks.value += chunks
            
    def get_stats(self):
        return {
            "processed": self.processed_count.value,
            "failed": self.failed_count.value,
            "total_chunks": self.total_chunks.value
        }

# Global stats object
stats = ProcessingStats()

class PubMedProcessor:
    def __init__(self, config: Dict = None):
        """Initialize the PubMed processor with configuration."""
        self.config = {
            "chunk_size": 350,
            "chunk_overlap": 0.15,
            "min_chunk_size": 100,
            "max_chunk_size": 500,
            "respect_paragraphs": True,
            "xml_namespaces": {"xlink": "http://www.w3.org/1999/xlink"},
            "output_dir": "./processed",
            "no_save": False,
            "batch_size": 100,
            "workers": 8
        }
        if config:
            self.config.update(config)
        
        # Create output directory if needed
        if "output_dir" in self.config and not self.config.get("no_save", False):
            os.makedirs(self.config["output_dir"], exist_ok=True)
            logger.info(f"Output directory: {self.config['output_dir']}")

    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\s*\d+\s*\]', '', text)
        return text.strip()

    def parse_file(self, file_path: str) -> Optional[etree._Element]:
        """Parse an XML file and return the root element."""
        try:
            parser = etree.XMLParser(remove_blank_text=True, recover=True)
            tree = etree.parse(file_path, parser)
            return tree.getroot()
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _get_text_content(self, element) -> str:
        """Extract text content from an XML element."""
        if element is None:
            return ""

        try:
            if not element.getchildren():
                return element.text or ""
        except AttributeError:
            # For newer lxml versions
            if len(element) == 0:
                return element.text or ""

        return "".join(element.itertext()).strip()

    def extract_metadata(self, root: etree._Element) -> Dict:
        """Extract metadata from the XML root element."""
        metadata = {
            "processing_date": datetime.now().isoformat()
        }

        def safe_extract(xpath, is_list=False, attribute=None, root_elem=root):
            try:
                elements = root_elem.xpath(xpath, namespaces=self.config["xml_namespaces"])
                if not elements:
                    return [] if is_list else None

                if is_list:
                    if attribute:
                        return [e.get(attribute) for e in elements if e.get(attribute)]
                    return [self._get_text_content(e) for e in elements]

                if attribute:
                    return elements[0].get(attribute)
                return self._get_text_content(elements[0])
            except Exception as e:
                logger.warning(f"Error extracting {xpath}: {e}")
                return [] if is_list else None

        # Extract basic metadata
        metadata["pmid"] = safe_extract('.//article-id[@pub-id-type="pmid"]')
        metadata["pmc"] = safe_extract('.//article-id[@pub-id-type="pmc"]')
        metadata["doi"] = safe_extract('.//article-id[@pub-id-type="doi"]')
        metadata["title"] = safe_extract('.//article-title')
        metadata["article_type"] = safe_extract('.//article/@article-type', attribute='article-type')
        metadata["journal"] = safe_extract('.//journal-title')
        metadata["issn"] = safe_extract('.//issn')
        metadata["volume"] = safe_extract('.//volume')
        metadata["issue"] = safe_extract('.//issue')

        # Extract publication date
        pub_year = safe_extract('.//pub-date/year')
        pub_month = safe_extract('.//pub-date/month')
        pub_day = safe_extract('.//pub-date/day')

        if pub_year:
            metadata["publication_date"] = pub_year
            if pub_month:
                if pub_day:
                    metadata["publication_date"] = f"{pub_year}-{pub_month}-{pub_day}"
                else:
                    metadata["publication_date"] = f"{pub_year}-{pub_month}"

        # Extract authors
        authors = []
        author_elements = root.xpath('.//contrib[@contrib-type="author"]')
        for author_elem in author_elements:
            author = {}
            surname = safe_extract('.//surname', root_elem=author_elem)
            given_names = safe_extract('.//given-names', root_elem=author_elem)

            if surname or given_names:
                if surname and given_names:
                    author["name"] = f"{surname}, {given_names}"
                else:
                    author["name"] = surname or given_names

                try:
                    aff_id = safe_extract('.//xref[@ref-type="aff"]/@rid', root_elem=author_elem)
                    if aff_id:
                        affiliation = safe_extract(f'.//aff[@id="{aff_id}"]')
                        if affiliation:
                            author["affiliation"] = affiliation
                except:
                    # Skip affiliation on error
                    pass

                authors.append(author)

        if authors:
            metadata["authors"] = authors

        # Extract keywords
        keywords = safe_extract('.//kwd', is_list=True)
        if keywords:
            metadata["keywords"] = keywords

        # Extract section titles
        sections = []
        section_titles = safe_extract('.//sec/title', is_list=True)
        if section_titles:
            metadata["sections"] = section_titles

        return metadata

    def extract_text_sections(self, root: etree._Element) -> List[Dict[str, str]]:
        """Extract text sections from the XML root element."""
        sections = []

        # Extract abstract
        abstract_elements = root.xpath('.//abstract')
        for abstract_elem in abstract_elements:
            title = "Abstract"
            abstract_sections = abstract_elem.xpath('.//sec')

            if abstract_sections:
                for sec in abstract_sections:
                    sec_title = sec.xpath('./title')
                    sec_text = "".join(sec.itertext()).strip()
                    if sec_title:
                        section_title = f"Abstract - {sec_title[0].text}"
                    else:
                        section_title = "Abstract"

                    sections.append({
                        "title": section_title,
                        "text": sec_text
                    })
            else:
                abstract_text = "".join(abstract_elem.itertext()).strip()
                if abstract_text:
                    sections.append({
                        "title": title,
                        "text": abstract_text
                    })

        # Extract body sections
        body_elements = root.xpath('.//body')
        for body_elem in body_elements:
            body_sections = body_elem.xpath('./sec')

            if body_sections:
                for sec in body_sections:
                    sec_title_elem = sec.xpath('./title')
                    if sec_title_elem:
                        sec_title = sec_title_elem[0].text
                    else:
                        sec_title = "Body"

                    if sec_title_elem:
                        sec_text = "".join(sec.itertext()).replace(sec_title, "", 1).strip()
                    else:
                        sec_text = "".join(sec.itertext()).strip()

                    sections.append({
                        "title": sec_title,
                        "text": sec_text
                    })
            else:
                body_text = "".join(body_elem.itertext()).strip()
                if body_text:
                    sections.append({
                        "title": "Body",
                        "text": body_text
                    })

        return sections

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text."""
        return len(text.split())

    def chunk_text(self, sections: List[Dict[str, str]], metadata: Dict) -> List[Dict]:
        """Split sections into chunks based on token size."""
        chunks = []
        chunk_id = 0

        for section in sections:
            section_title = section["title"]
            section_text = self.normalize_text(section["text"])

            if not section_text:
                continue

            # Split by paragraphs if enabled
            if self.config["respect_paragraphs"]:
                paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', section_text)
            else:
                paragraphs = [section_text]

            current_chunk = ""
            current_tokens = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                paragraph_tokens = self.estimate_tokens(paragraph)

                if paragraph_tokens > self.config["max_chunk_size"]:
                    if current_tokens > 0:
                        chunk = {
                            "chunk_id": f"{metadata.get('pmid', 'unknown')}-{chunk_id}",
                            "section": section_title,
                            "text": current_chunk,
                            "token_count": current_tokens
                        }
                        chunks.append(chunk)
                        chunk_id += 1
                        current_chunk = ""
                        current_tokens = 0

                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    temp_tokens = 0

                    for sentence in sentences:
                        sentence_tokens = self.estimate_tokens(sentence)

                        if temp_tokens + sentence_tokens <= self.config["chunk_size"]:
                            if temp_chunk:
                                temp_chunk += " " + sentence
                            else:
                                temp_chunk = sentence
                            temp_tokens += sentence_tokens
                        else:
                            if temp_chunk:
                                chunk = {
                                    "chunk_id": f"{metadata.get('pmid', 'unknown')}-{chunk_id}",
                                    "section": section_title,
                                    "text": temp_chunk,
                                    "token_count": temp_tokens
                                }
                                chunks.append(chunk)
                                chunk_id += 1

                            temp_chunk = sentence
                            temp_tokens = sentence_tokens

                    if temp_chunk:
                        chunk = {
                            "chunk_id": f"{metadata.get('pmid', 'unknown')}-{chunk_id}",
                            "section": section_title,
                            "text": temp_chunk,
                            "token_count": temp_tokens
                        }
                        chunks.append(chunk)
                        chunk_id += 1

                elif current_tokens + paragraph_tokens <= self.config["chunk_size"]:
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
                else:
                    chunk = {
                        "chunk_id": f"{metadata.get('pmid', 'unknown')}-{chunk_id}",
                        "section": section_title,
                        "text": current_chunk,
                        "token_count": current_tokens
                    }
                    chunks.append(chunk)
                    chunk_id += 1

                    current_chunk = paragraph
                    current_tokens = paragraph_tokens

            if current_chunk and current_tokens >= self.config["min_chunk_size"]:
                chunk = {
                    "chunk_id": f"{metadata.get('pmid', 'unknown')}-{chunk_id}",
                    "section": section_title,
                    "text": current_chunk,
                    "token_count": current_tokens
                }
                chunks.append(chunk)
                chunk_id += 1

        # Add metadata to each chunk
        for chunk in chunks:
            chunk["metadata"] = {
                "pmid": metadata.get("pmid"),
                "title": metadata.get("title"),
                "journal": metadata.get("journal"),
                "publication_date": metadata.get("publication_date"),
                "doi": metadata.get("doi")
            }
            # Remove None values
            chunk["metadata"] = {k: v for k, v in chunk["metadata"].items() if v is not None}

        return chunks

    def process_file(self, file_path: str) -> Optional[Dict]:
        """Process a single XML file."""
        try:
            # Parse the file
            root = self.parse_file(file_path)
            if root is None:
                logger.error(f"Failed to parse {file_path}")
                stats.update(failed=1)
                return None

            # Extract metadata
            metadata = self.extract_metadata(root)
            if not metadata.get("pmid"):
                metadata["pmid"] = Path(file_path).stem

            # Extract text sections
            sections = self.extract_text_sections(root)

            # Chunk text
            chunks = self.chunk_text(sections, metadata)

            # Create document
            document = {
                "document_id": metadata.get("pmid", Path(file_path).stem),
                "file_path": str(file_path),
                "metadata": metadata,
                "chunks": chunks,
                "chunk_count": len(chunks)
            }

            # Update stats
            stats.update(processed=1, chunks=len(chunks))
            
            # Log progress
            current_stats = stats.get_stats()
            logger.info(f"Processed {file_path}: {len(sections)} sections, {len(chunks)} chunks | "
                       f"Total: {current_stats['processed']} processed, {current_stats['failed']} failed, "
                       f"{current_stats['total_chunks']} chunks")
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            stats.update(failed=1)
            return None

    def save_document(self, document: Dict, output_format: str = "json") -> str:
        """Save the processed document and return the output path."""
        if self.config.get("no_save", False):
            return document["document_id"]
            
        output_dir = self.config.get("output_dir", ".")
        document_id = document["document_id"]

        if output_format == "json":
            output_file = os.path.join(output_dir, f"{document_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved to {output_file}")
            return output_file
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return ""

    def process_batch(self, batch_files: List[str], batch_id: int) -> Tuple[List[Dict], List[str]]:
        """Process a batch of files and return the documents."""
        logger.info(f"Starting batch {batch_id} with {len(batch_files)} files")
        batch_start_time = datetime.now()
        
        documents = []
        processed_files = []
        
        for file_path in batch_files:
            document = self.process_file(file_path)
            if document:
                documents.append(document)
                if not self.config.get("no_save", False):
                    output_file = self.save_document(document)
                    processed_files.append(output_file)
                else:
                    processed_files.append(document["document_id"])
        
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        logger.info(f"Completed batch {batch_id}: {len(documents)} documents in {batch_duration:.2f} seconds")
        
        return documents, processed_files

    def worker_process_file(self, file_path: str) -> Optional[Dict]:
        """Worker function for multiprocessing."""
        return self.process_file(file_path)
        
    def process_directory(self, directory_path: str, output_format: str = "json", file_limit: int = None) -> List[Dict]:
        """Process all XML files in a directory using batch processing and multiprocessing."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []

        # Find all XML files
        all_files = list(directory.glob("**/*.xml")) + list(directory.glob("**/*.nxml"))
        all_files = [str(f) for f in all_files]
        
        total_files = len(all_files)
        logger.info(f"Found {total_files} XML files in {directory_path}")

        # Limit the number of files if specified
        if file_limit and file_limit > 0:
            files_to_process = all_files[:file_limit]
            logger.info(f"Processing {len(files_to_process)} out of {total_files} files")
        else:
            files_to_process = all_files

        # Process in batches with multiprocessing
        batch_size = self.config.get("batch_size", 100)
        num_workers = min(self.config.get("workers", 8), multiprocessing.cpu_count())
        
        # Create batches
        batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
        logger.info(f"Created {len(batches)} batches with batch size {batch_size}")
        logger.info(f"Using {num_workers} worker processes")
        
        all_documents = []
        
        # Use multiprocessing for each batch
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(self.worker_process_file, files_to_process),
                total=len(files_to_process),
                desc=f"Processing {total_files} files"
            ))
            
            # Filter out None results
            valid_results = [doc for doc in results if doc is not None]
            all_documents.extend(valid_results)
            
            # Save documents if not disabled
            if not self.config.get("no_save", False):
                for doc in valid_results:
                    self.save_document(doc, output_format)
        
        # Log final statistics
        final_stats = stats.get_stats()
        logger.info(f"Processing complete. Successfully processed: {final_stats['processed']}, "
                   f"Failed: {final_stats['failed']}, Total chunks: {final_stats['total_chunks']}")
                   
        logger.info(f"Average chunks per document: {final_stats['total_chunks'] / max(1, final_stats['processed']):.2f}")
        
        return all_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse PubMed XML files and chunk text")
    parser.add_argument("--input", "-i", required=True, help="Directory containing XML files")
    parser.add_argument("--output", "-o", default="./processed", help="Output directory for processed files")
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of files to process")
    parser.add_argument("--chunk-size", type=int, default=350, help="Target chunk size in tokens")
    parser.add_argument("--min-chunk-size", type=int, default=100, help="Minimum chunk size in tokens")
    parser.add_argument("--max-chunk-size", type=int, default=500, help="Maximum chunk size in tokens")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of files to process in each batch")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--no-save", action="store_true", help="Don't save intermediate files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main function to run the parser."""
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create config from arguments
    config = {
        "output_dir": args.output,
        "chunk_size": args.chunk_size,
        "min_chunk_size": args.min_chunk_size,
        "max_chunk_size": args.max_chunk_size,
        "no_save": args.no_save,
        "batch_size": args.batch_size,
        "workers": args.workers
    }
    
    # Create processor and process files
    processor = PubMedProcessor(config)
    processed_documents = processor.process_directory(args.input, file_limit=args.limit)
    
    # Return the processed documents for pipeline use
    return processed_documents


if __name__ == "__main__":
    main()