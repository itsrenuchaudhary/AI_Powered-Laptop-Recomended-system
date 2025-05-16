# 💻 AI Laptop Recommendation Assistant

An intelligent AI-powered assistant that helps users find the best laptops based on their needs, budget, and preferences. This system uses LangChain, FAISS, HuggingFace embeddings, and Ollama (LLaMA3) to deliver contextual, relevant, and personalized recommendations through a Gradio interface.

---

## 🔍 Features

* Conversational product search for laptops
* Filters results by price range and RAM from user queries
* Retrieves laptops using a RAG pipeline (FAISS + LLM)
* Supports intelligent follow-up questions
* Keeps conversational memory across interactions
* Clean and structured product recommendations
* Available in both standard and chatbot UI (Gradio)

---

## 🧠 Technologies Used

* Python
* LangChain (chains, retrievers, memory, etc.)
* HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* FAISS (vector database for semantic search)
* Ollama (serving LLaMA3 locally)
* Gradio (UI interface)
* Pandas (for CSV processing)
* Regular Expressions (for filtering logic)

---

## 🗂 Project Structure

```
├── data/
│   └── cleaned_laptops_data.csv       # Cleaned laptop product dataset
├── faiss_index/                       # Saved FAISS vectorstore
├── connect_with_llm.py               # Core logic for RAG pipeline and follow-up handling
├── main_app.py                        # Gradio chatbot interface
├── README.md                          # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/laptop-recommender-ai.git
cd laptop-recommender-ai
```

### 2. Install Dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

Sample dependencies (put in requirements.txt):

```txt
langchain
langchain-community
langchain-core
langchain-huggingface
sentence-transformers
faiss-cpu
pandas
gradio
ollama
```

### 3. Download & Prepare Data

Place your laptop dataset as a CSV in the data/ folder:

* Filename: cleaned\_laptops\_data.csv
* Columns: should include Product Name, Price, RAM, and key specs as plain text.

### 4. Run the Embedding Pipeline (One-time)

```bash
python connect_with_llm.py
```

This will:

* Load the dataset
* Split text into chunks
* Embed it using HuggingFace embeddings
* Save a FAISS vector index to disk

### 5. Launch the App

```bash
python main_app.py
```

---

## 🧠 How It Works

* User submits a query like: “Best laptops under ₹60000 with 8GB RAM”
* FAISS + Embeddings perform semantic search on laptop documents
* Filter logic extracts price/RAM constraints from query using regex
* A LLaMA3 model (via Ollama) is used to generate product responses from context
* Gradio interface displays results and handles follow-up questions

---

## 💬 Example Queries

* Best laptops under ₹50000
* I need a gaming laptop with 16GB RAM
* Which one is better for office work?
* Any options with Ryzen 7?

---

## ✅ To Do / Improvements

* ✅ Add price and RAM filtering
* ✅ Enable follow-up Q\&A from user
* ✅ Clean up duplicate results
* 🔄 Add GPU filter / storage filtering
* 🔄 Streamlit integration (optional)
* 🔄 Add product links or images

---

## 📸 Screenshot

<!-- Add a screenshot of the app UI here if available -->

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

Krishn Chaudhary
Pune, India
🔍 .NET Developer | AI Enthusiast | Open Source Contributor

Feel free to connect with me on [LinkedIn](https://linkedin.com/in/your-profile) or reach out for collaborations!

---

Let me know if you want this in markdown format or want me to generate a downloadable file.


```

---

Let me know if you'd like a `requirements.txt` or a sample `cleaned_laptops_data.csv` structure added too.
```
