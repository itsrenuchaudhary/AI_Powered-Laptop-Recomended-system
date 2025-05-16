import re
import gradio as gr
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

FAISS_INDEX_DIR = "faiss_index"

def get_session_history(session_id: str):
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")

chat_history = []
last_context = ""

def is_laptop_query(query):
    keywords = ["laptop", "notebook", "macbook", "ultrabook", "chromebook", "gaming laptop"]
    query = query.lower()
    return any(keyword in query for keyword in keywords)

def extract_price(text):
    text = text.replace('\ufeff', '').replace('â‚¹', '₹').replace('Rs.', '₹').replace('INR', '₹')
    match = re.search(r"₹\s?(\d{1,3}(?:,\d{3})*|\d+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None

def extract_ram(text):
    match = re.search(r"(\d{1,2})\s*GB\s*(?:DDR\d)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def filter_by_price(documents, query):
    query = query.lower()
    between = re.search(r"between\s*₹?(\d+)\s*(?:and|-)\s*₹?(\d+)", query) or \
              re.search(r"range\s*₹?(\d+)\s*(?:to|-)\s*₹?(\d+)", query)
    above = re.search(r"(?:above|over|greater than|more than)\s*₹?(\d+)", query)
    below = re.search(r"(?:below|under|less than)\s*₹?(\d+)", query)
    ram_match = re.search(r"(\d{1,2})\s*gb\s*ram", query)
    required_ram = int(ram_match.group(1)) if ram_match else None

    filtered_docs = []
    for doc in documents:
        text = doc.page_content
        price = extract_price(text)
        ram = extract_ram(text)
        if price is None:
            continue

        passes_price = (between and int(between.group(1)) <= price <= int(between.group(2))) or \
                       (above and price > int(above.group(1))) or \
                       (below and price < int(below.group(1))) or \
                       (not (between or above or below))
        passes_ram = ram is not None and (required_ram is None or ram >= required_ram)
        if passes_price and passes_ram:
            filtered_docs.append(doc)
    return filtered_docs

def clean_response(text):
    text = re.sub(r"(🔍 Top 5 Recommended Products:\s*)+", "🔍 Top 5 Recommended Products:\n", text)
    lines = text.strip().split('\n')
    seen_titles = set()
    cleaned = []

    for line in lines:
        if "**Product Name**" in line:
            product_name = line.lower()
            if product_name in seen_titles:
                continue
            seen_titles.add(product_name)
        cleaned.append(line)

    return '\n'.join(cleaned)


# Load models and prompt
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(FAISS_INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})
llm = Ollama(model="llama3")

main_prompt = PromptTemplate(
    template="""
You are a professional product assistant helping users select the best laptops.

Instructions:
- ONLY use the given context for your answers.
- DO NOT add any assumptions or external data.
- Select the TOP 5 laptops based on the user's query.
- Present the result in a clear, structured format using bullet points.
- Keep the language concise and readable.

Output Format:
🔍 Top 5 Recommended Products:

1. **Product Name**: <Name>
   - **Price**: ₹<price>
   - **Key Features**: <summarize in one line>
   - **Reason for Recommendation**: <why it's a good fit>

Context:
{context}

User Query:
{question}

Response:
""",
    input_variables=["context", "question"]
)

# main_chain = LLMChain(llm=llm, prompt=main_prompt)
main_chain = main_prompt | llm


followup_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Based on the following product recommendations, answer the user's follow-up question.

Context:
{context}

Follow-up Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# followup_chain = LLMChain(llm=llm, prompt=followup_prompt)
followup_chain = followup_prompt | llm

# History wrapper only on main_chain
conversational_chain = RunnableWithMessageHistory(
    main_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Handle main query
def handle_query(user_query):
    global chat_history, last_context

    # chat_history.append(("user", user_query))

    if not is_laptop_query(user_query):
        response = "⚠️ Please ask only about laptops. Queries related to other products are not supported."
        chat_history.append(("assistant", response))
        return response

    chat_history.append(("user", user_query))
    retrieved_docs = retriever.get_relevant_documents(user_query)
    filtered_docs = filter_by_price(retrieved_docs, user_query)

    if not filtered_docs:
        response = "⚠️ Sorry, no laptops found matching your criteria."
    else:
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        last_context = context
        response = main_chain.invoke({"context": context, "question": user_query})
        response = clean_response(response)

    chat_history.append(("assistant", response))
    return response

# Handle follow-up
def handle_followup(followup_question):
    global chat_history, last_context


    if not last_context:
        response = "⚠️ Please ask a main query first to get recommendations."
        chat_history.append(("user", followup_question))
        chat_history.append(("assistant", response))
        return response
    
    chat_history.append(("user", followup_question))
   
    response = followup_chain.invoke({
        "context": last_context,
        "question": followup_question
    })

    chat_history.append(("assistant", response))
    return response



def load_resources():
    return retriever, conversational_chain



# Optional: Display full chat history
def get_full_chat():
    return "\n".join([f"🧑 {m}" if r == "user" else f"🤖 {m}" for r, m in chat_history])

# Gradio interface
with gr.Blocks(title="AI Laptop Recommender") as demo:
    gr.Markdown("# 💻 AI Laptop Recommendation Assistant")

    with gr.Row():
        main_query = gr.Textbox(label="Enter your laptop requirement", placeholder="e.g. Laptop under ₹80000 with 16GB RAM")
        submit_btn = gr.Button("🔍 Get Recommendations")
    result_output = gr.Textbox(label="Product Recommendations", lines=15)

    with gr.Row():
        followup_input = gr.Textbox(label="Follow-up Question", placeholder="e.g. Which one is best for office work?")
        followup_btn = gr.Button("💬 Ask Follow-up")
    followup_output = gr.Textbox(label="Assistant Answer", lines=5)

    with gr.Row():
        chat_btn = gr.Button("📜 View Chat History")
        chat_display = gr.Textbox(label="Full Conversation", lines=20)

    submit_btn.click(fn=handle_query, inputs=main_query, outputs=result_output)
    followup_btn.click(fn=handle_followup, inputs=followup_input, outputs=followup_output)
    chat_btn.click(fn=get_full_chat, outputs=chat_display)

demo.launch()














