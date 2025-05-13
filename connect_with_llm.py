import os
import csv
import json
import re
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

# Extract ₹ amount
def extract_budget(query):
    match = re.search(r"₹(\d+[\d,]*)", query)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

def extract_ram(text):
    match = re.search(r"(\d{1,2})\s*GB\s*(?:DDR\d)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None




def filter_by_price(documents, query):
    query = query.lower()
    
    # Extracting price range
    between = re.search(r"between\s*₹?(\d+)\s*(?:and|-)\s*₹?(\d+)", query) or \
              re.search(r"range\s*₹?(\d+)\s*(?:to|-)\s*₹?(\d+)", query)
    above = re.search(r"(?:above|over|greater than|more than)\s*₹?(\d+)", query)
    below = re.search(r"(?:below|under|less than)\s*₹?(\d+)", query)
    
    # Extract RAM requirement from query
    ram_match = re.search(r"(\d{1,2})\s*gb\s*ram", query)
    required_ram = int(ram_match.group(1)) if ram_match else None

    filtered_docs = []
    for doc in documents:
        text = doc.page_content
        price = extract_price(text)
        ram = extract_ram(text)

        print(f"Extracted price from document: {price}")
        print(f"Extracted RAM from document: {ram}")

        if price:
            passes_price_filter = False
            if between:
                low, high = int(between.group(1)), int(between.group(2))
                passes_price_filter = low <= price <= high
            elif above:
                value = int(above.group(1))
                passes_price_filter = price > value
            elif below:
                value = int(below.group(1))
                passes_price_filter = price < value
            else:
                passes_price_filter = True  # No price constraint

            # RAM filter
            passes_ram_filter = True
            if required_ram is not None and ram is not None:
                passes_ram_filter = ram == required_ram

            if passes_price_filter and passes_ram_filter:
                filtered_docs.append(doc)

    return filtered_docs


def extract_price(text):
    # Fix encoding issues and normalize
    text = text.replace('\ufeff', '').replace('â‚¹', '₹').replace('Rs.', '₹').replace('INR', '₹')

    # Look for ₹ followed by digits (with optional commas)
    match = re.search(r"₹\s?(\d{1,3}(?:,\d{3})+|\d+)", text)
    if match:
        extracted = match.group(1).replace(",", "")
        print(f"Extracted price: {extracted}")
        return int(extracted)
    return None


def clean_response(text):
    text = re.sub(r"(🔍 Top 2 Recommended Products:\s*)+", "🔍 Top 2 Recommended Products:\n", text)
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



# Load Ollama LLM
def load_llm_ollama(model_name="llama3"):
    return Ollama(model=model_name)


# Load embedding and FAISS index
llm = load_llm_ollama("llama3")
DB_FAISS_PATH = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 15})

# Memory for context tracking
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Log clicks
def log_behavior(query, clicked_products, filepath="user_behavior_log.csv"):
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for product in clicked_products:
            writer.writerow([query, product])



# Custom prompt for top 2 recommendations
CUSTOM_PROMPT_TEMPLATE = """
You are a professional product assistant helping users select the best laptops.

Instructions:
- ONLY use the given context for your answers.
- DO NOT add any assumptions or external data.
- Select the TOP 2 laptops based on the user's query.
- Present the result in a clear, structured format using bullet points.
- Keep the language concise and readable.

Output Format:
🔍 Top 2 Recommended Products:

1. **Product Name**: <Name>
   - **Price**: ₹<price>
   - **Key Features**: <summarize in one line>
   - **Reason for Recommendation**: <why it's a good fit>

   

2. **Product Name**: <Name>
   - **Price**: ₹<price>
   - **Key Features**: <summarize in one line>
   - **Reason for Recommendation**: <why it's a good fit>

Context:
{context}

User Query:
{question}

Response:
"""


FOLLOWUP_PROMPT = """
You are a helpful assistant who answers follow-up questions about selected laptops.

Only use the following product details to answer the user's question.

Product Info:
{context}

User Question:
{question}

Answer:
"""




# Prompt functions
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Main CLI loop
while True:
    user_query = input("\n📝 Your query (type 'exit' to stop): ")
    if user_query.lower() == "exit":
        break

    memory.save_context({"input": user_query}, {"output": "Query submitted"})

    # Retrieve and filter
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    filtered_docs = filter_by_price(retrieved_docs, user_query)

    print("\n📦 Filtered Products Based on Price and RAM:\n")
    for i, doc in enumerate(filtered_docs, 1):
        print(f"{i}. {doc.page_content.strip()[:300]}...\n")

    if not filtered_docs:
        print("⚠️ No products found within the specified price range.")
        continue

    # Ask for selected product numbers
    selected_indexes = input("👉 Enter the product numbers you liked (comma-separated): ")
    selected_indexes = [int(idx.strip()) for idx in selected_indexes.split(',') if idx.strip().isdigit()]
    clicked_products = [filtered_docs[i-1].metadata.get("source", f"Product {i}") for i in selected_indexes]

    log_behavior(user_query, clicked_products)

    memory.save_context(
        {"input": f"User clicked on: {', '.join(clicked_products)}"},
        {"output": "Click logged"}
    )

    # Run recommendation
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": user_query})
    print("\n🔍 Recommended Products:\n")
    print(clean_response(response))

    # Get selected product docs for follow-up
    selected_product_docs = [filtered_docs[i-1] for i in selected_indexes]
    followup_context = "\n\n".join([doc.page_content for doc in selected_product_docs])

    # Follow-up chain
    followup_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["context", "question"], template=FOLLOWUP_PROMPT)
    )

    # Handle follow-up questions
    while True:
        followup_question = input("\n🤖 Ask a follow-up question about the selected laptops (or type 'no' to continue): ")
        if followup_question.lower() in ["no", "n", "exit"]:
            break

        followup_response = followup_chain.run(context=followup_context, question=followup_question)
        print(f"\n💬 Answer: {followup_response}")

    # Show memory
    print("\n🧠 Memory State:\n")
    print(memory.buffer)






















