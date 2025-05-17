# 💻 AI Laptop Recommendation Assistant

An intelligent AI-powered assistant that helps users find the best laptops based on their needs, budget, and preferences. This system uses LangChain, FAISS, HuggingFace embeddings, and RAG, Ollama (LLaMA3) to deliver contextual, relevant, and personalized recommendations through a Gradio interface.

---

## 🚀 Features

* 🔍 Intelligent **product search** from Flipkart data
* 🧠 RAG-based query answering using **FAISS** and **LLM (LLaMA3)**
* 💬 Conversational follow-up support with **chat history**
* 🧾 Filters laptops by **price range** and **RAM size**
* 🖥️ Easy-to-use **Gradio web interface**

---


## 🧠 Technologies Used

* Python
* LangChain (chains, retrievers, memory, etc.)
* HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
* FAISS (vector database for semantic search)
* RAG (Retrive data)
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


---




## 📂 Project Structure

```bash
├── data/
│   └── cleaned_laptops_data.csv     # Preprocessed laptop listings
├── faiss_index/                     # Saved FAISS index
├── main_app.py                      # Gradio app (chatbot interface)
├── connect_with_llm.py             # Core logic: retrieval, filtering, LLM invocation
├── flipkart_scraper.py             # (Optional) Web scraper for Flipkart laptops
├── requirements.txt
└── README.md
```

---

## 📈 How it Works

1. **Scrape Flipkart Laptop Data** (optional):

   * Scrape product name, price, description, and rating from Flipkart using BeautifulSoup.
   * Save it as a CSV.

2. **Vector Index Creation**:

   * Load laptop data using `CSVLoader`.
   * Split into chunks with `RecursiveCharacterTextSplitter`.
   * Embed with HuggingFace `MiniLM` and index using FAISS.

3. **User Query Handling**:

   * Parse user query for laptop-related terms, price range, and RAM constraints.
   * Filter relevant documents from FAISS index.
   * Use `LLaMA3` via `Ollama` to generate top 5 recommendations based on a structured prompt.

4. **Follow-up Support**:

   * Users can ask contextual follow-ups.
   * Memory is managed using `ConversationBufferMemory`.



---



## ✅ Example Queries

* `"Best laptops under ₹60000 with 16GB RAM"`
* `"Gaming laptops over ₹70000"`
* `"I need a laptop for office use under ₹50000"`
* `"Which one is best for students?"` *(follow-up)*

---


## 🧠 Future Improvements

* ✅ Add GPU/Processor filters
* ✅ Include real-time web scraping updates
* 🚧 Integrate multi-turn memory with document feedback
* 🚧 Add image-based comparison view






