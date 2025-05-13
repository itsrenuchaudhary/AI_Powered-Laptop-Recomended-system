import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load CSV
DATA_PATH = "data/cleaned_laptops_data.csv"  # make sure it's in the same folder or give full path

loader = CSVLoader(file_path=DATA_PATH)
documents = loader.load()

print("Number of documents loaded:", len(documents))

# Step 2: Split text (optional if needed)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print("Number of doc loaded:", len(docs))
# Step 3: Create embeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create and save FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save FAISS index to disk
db.save_local("faiss_index")
print("FAISS index saved.")


