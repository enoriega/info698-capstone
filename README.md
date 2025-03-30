# INFO698 Capstone - Agent RAG

This repository contains the implementation of a ** RAG (Retrieval-Augmented Generation)** pipeline for PubMedCentral Database.

---

## 🛠️ Setup (Ubuntu)

### 1️⃣ Update System and Install Python 3.12
```bash
sudo apt update
sudo apt install python3.12-venv
sudo apt install python3.12
```

### 2️⃣ Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install UV (Ultra Fast Python Package Manager)
```bash
pip install uv
```

### 4️⃣ Install Project Dependencies
```bash
uv pip install -r pyproject.toml
```

---

## 🚀 Running the App

### Launch Streamlit UI
```bash
streamlit run main.py
```

This will start a local Streamlit app where you can input your biomedical queries and get subqueries + answers.

---
