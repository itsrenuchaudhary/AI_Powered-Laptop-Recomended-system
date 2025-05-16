

---

```markdown
# 💻 AI Laptop Recommendation Assistant

An AI-powered conversational assistant that helps users find the best laptops based on their requirements such as price, RAM, and usage needs (e.g., gaming, office work). It supports follow-up questions and remembers previous context for a personalized experience.

## 🔍 Features

- Conversational product search assistant for laptops
- Filters by price (under, above, between ranges) and RAM
- Retrieves relevant products using semantic search (FAISS + HuggingFace embeddings)
- Follow-up question support with contextual awareness
- Dual interface: Structured form (Gradio Blocks) and conversational chatbot (Gradio Chatbot)
- Local language model support via Ollama (LLaMA3)
- Memory-enabled dialogue using LangChain

---

## 🛠️ Technologies Used

- **LangChain**: Conversational chains, memory, and vector store integrations
- **FAISS**: Fast vector search for similarity-based document retrieval
- **HuggingFace Embeddings**: For semantic similarity
- **Gradio**: User-friendly front-end interface
- **Ollama**: To serve local LLMs like `llama3`
- **Pandas & CSVLoader**: For ingesting and processing laptop data

---

## 📦 Folder Structure

```

.
├── data/
│   └── cleaned\_laptops\_data.csv         # Laptop dataset
├── faiss\_index/                         # Saved FAISS vector index
├── connect\_with\_llm.py                 # Core logic: loading, filtering, chains
├── main\_app.py                         # Chatbot-based Gradio interface
├── app\_with\_blocks.py                  # Form-based Gradio interface
└── README.md

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-laptop-assistant.git
cd ai-laptop-assistant
````

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Start Ollama with LLaMA3 Model

Make sure Ollama is installed and running:

```bash
ollama run llama3
```

### 4. Prepare Data & Build FAISS Index

Run this once to create the vector index:

```bash
python connect_with_llm.py
```

This loads the laptop data, splits documents, generates embeddings, and stores the FAISS index.

### 5. Launch the App

#### Option A: Chatbot Interface

```bash
python main_app.py
```

#### Option B: Form-based Interface with Follow-up Support

```bash
python app_with_blocks.py
```

---

## 💡 Example Queries

* "Best laptops under ₹60000 with 8GB RAM"
* "Suggest a gaming laptop between ₹70000 and ₹90000"
* "Above ₹80000 with at least 16GB RAM"
* Follow-up: "Which one is light and good for travel?"

---

## 🧠 How It Works

1. **User Query** → Parsed for keywords like price, RAM
2. **Retriever** → FAISS fetches relevant laptop entries
3. **Filter** → Additional filtering applied on price/RAM
4. **LLM Prompt** → LLaMA3 generates top 5 recommendations
5. **Follow-up** → Context-aware LLM answers additional questions

---

## ✅ To-Do / Improvements

* ✅ Add price and RAM filtering logic
* ✅ Implement follow-up question handling
* ✅ Contextual memory via LangChain
* 🔄 Add GPU or battery filtering
* 🔄 Improve UI layout and usability
* 🔄 Export recommendations as PDF

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.com)
* [Gradio](https://www.gradio.app/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [HuggingFace Sentence Transformers](https://www.sbert.net/)

```

---

Let me know if you'd like a `requirements.txt` or a sample `cleaned_laptops_data.csv` structure added too.
```

