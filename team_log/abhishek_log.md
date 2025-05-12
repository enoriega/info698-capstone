# Create the markdown content for Abhishek's contribution log
md_content = """# Capstone Project: Abhishek’s Contribution Log

**Contributor:** Abhishek Kumar  
**Role:** PubMed XML Processing & Vector Embedding Pipeline  
**Timeline:** February 24, 2025 – May 7, 2025

---

### Week of Feb 24, 2025  
**Project Planning and Setup**  
- Attended initial meetings to understand project scope and architecture.  
- Assigned responsibility for PubMed XML intake, metadata extraction, text chunking, and embedding preparation.  
- Explored Python tools and libraries for parsing and data processing (e.g., `lxml`, `SentenceTransformers`).  

---

### Week of Mar 3, 2025  
**XML Parsing and Metadata Extraction**  
- Implemented XML parser using `lxml.etree` for memory-efficient processing.  
- Extracted metadata (PMID, title, abstract, MeSH terms) using optimized XPath expressions.  
- Created scripts to validate metadata extraction against a ground truth sample.  

---

### Week of Mar 10, 2025 – Spring Break Week  
**Research and Review**  
- Reviewed chunking strategies and tokenization techniques.  
- Discussed overlap percentage and chunk boundary handling with the team.  

---

### Week of Mar 17, 2025  
**Chunking Design and Implementation**  
- Designed and implemented semantic text chunking logic.  
- Experimented with 200–400 token chunks and overlap of 10–20%.  
- Ensured boundary alignment to avoid splitting of key biomedical terms.  

---

### Week of Mar 24, 2025  
**Normalization and Preprocessing**  
- Added Unicode normalization (NFC/NFKC) and whitespace cleanup functions.  
- Benchmarked different normalization strategies and finalized the approach.  

---

### Week of Mar 31, 2025  
**Pipeline Integration and Testing**  
- Developed chunk-to-document mapping and metadata tagging logic.  
- Structured output for downstream embedding generation.  
- Collaborated with the team to align input/output schemas.  

---

### Week of Apr 7, 2025  
**Weaviate Integration**  
- Helped migrate chunked data and metadata to Weaviate database.  
- Tested indexing performance and semantic retrieval queries using Weaviate Python client.  

---

### Week of Apr 14, 2025  
**Embedding Generation and Model Benchmarking**  
- Integrated `BGE-large-en-v1.5`, `PubMedBERT`, and `MiniLM` for embedding generation.  
- Benchmarked embeddings using cosine similarity and clustering metrics.  
- Prepared vector output for semantic search using Weaviate.

---

### Week of Apr 21, 2025  
**Final Integration and Cleanup**  
- Finalized pipeline components.  
- Validated input-output consistency across modules.  
- Assisted with documentation and shared module walkthrough.

---

### Week of Apr 28, 2025  
**Report and Poster Finalization**  
- Completed technical write-up for the pipeline.  
- Contributed to the final project poster design and reviewed team submissions.

---

### Week of May 5, 2025  
**Project Presentation**  
- Final debugging and deployment checks.  
- Presented project at iShowcase event on May 7, 2025.
"""

# Save to .md file
md_file_path = "/mnt/data/Abhishek_Contribution_Log.md"
with open(md_file_path, "w") as f:
    f.write(md_content)

md_file_path
