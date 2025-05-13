# 🛒 AI-Powered Laptop Recommendation Assistant (Flipkart Scraper + RAG + Ollama)

This is a terminal-based AI-powered product assistant that scrapes **laptop data from Flipkart**, builds a **vector store using FAISS**, and enables **conversational product recommendations** using **LLMs (LLaMA3 via Ollama)**. It supports **dynamic filtering** based on user inputs like price and RAM, maintains **conversational memory**, and handles **follow-up questions** on selected products.

---

## 📌 Features

* ✅ Scrapes real-time laptop listings from Flipkart using `BeautifulSoup`
* ✅ Extracts product name, price, description, and rating
* ✅ Converts CSV data into FAISS vector embeddings using HuggingFace
* ✅ Filters results based on price range (`under ₹50000`, `between ₹60000 and ₹70000`, etc.)
* ✅ RAM filtering (e.g., `at least 8GB RAM`)
* ✅ Provides top 2 AI-generated product recommendations
* ✅ Logs user clicks for behavior analysis
* ✅ Supports intelligent follow-up questions
* ✅ Uses conversational memory (`BufferMemory`) for context-aware interactions

---

## 🧠 Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* Ollama (LLaMA3)
* BeautifulSoup
* Pandas


---

## ⚙️ Setup Instructions

### . Install Dependencies

Use `pip` and create a virtual environment:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
pandas
requests
beautifulsoup4
lxml
langchain
faiss-cpu
sentence-transformers
ollama
```

> ⚠️ Install Ollama and run a local LLaMA3 model:

```bash
ollama run llama3
```

---
You’ll be prompted to enter queries like:

```
📝 Your query: laptops under ₹50000 with at least 8GB RAM
```

You'll get AI-curated recommendations and can interact via follow-up questions like:

```
🤖 How is the battery backup of the second one?
```

---

## 🧠 Example Queries

* `"laptops under ₹50000 with SSD"`
* `"between ₹60000 and ₹80000 with 16GB RAM"`
* `"best laptop for programming and ML"`
* `"which one has better display?"` *(as a follow-up)*

---

## 📊 Click Tracking

User selections are logged in `user_behavior_log.csv` with the original query and product clicked.

---

## 🛠 Future Improvements

* Integrate image previews in GUI (e.g., Streamlit)
* Add filters for CPU/GPU/brand
* Enable feedback loop for learning preferences
* Store chat memory persistently

---

## 🙌 Credits

* Built with [LangChain](https://www.langchain.com/)
* Embeddings by [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
* Local LLM via [Ollama](https://ollama.ai/)
* Data scraped from [Flipkart](https://www.flipkart.com/)

