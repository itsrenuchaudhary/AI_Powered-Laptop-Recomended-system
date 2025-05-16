# main_app.py

import gradio as gr
from connect_with_llm import handle_query, handle_followup, chat_history

# Global chat history for Gradio chatbot display
chat_display_history = []

def respond(user_input, history):
    global chat_display_history

    # Detect if it's a new laptop query or follow-up
    laptop_keywords = ["laptop", "notebook", "macbook", "chromebook", "gaming", "ultrabook"]
    is_new_query = any(word in user_input.lower() for word in laptop_keywords)

    if is_new_query:
        response = handle_query(user_input)
    else:
        if not chat_display_history:
            response = "⚠️ Please start with a laptop-related question."
        else:
            response = handle_followup(user_input)

    chat_display_history.append((user_input, response))
    return chat_display_history, chat_display_history

def clear_chat():
    global chat_display_history
    chat_display_history = []
    return [], ""

def show_full_chat():
    return "\n".join([f"🧑 {m}" if r == "user" else f"🤖 {m}" for r, m in chat_history])

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 💻 AI Laptop Assistant")
    gr.Markdown("Ask about laptops with price, RAM, or usage needs!")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something like: Best laptops under ₹60000 with 8GB RAM", label="Your query")

    with gr.Row():
        clear = gr.Button("🧹 Clear chat")
        show_history = gr.Button("📜 Show Internal Chat History")

    history_box = gr.Textbox(label="Logged chat_history (user + assistant)", lines=20)

    # Bind actions
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
    clear.click(clear_chat, outputs=[chatbot, msg])
    show_history.click(show_full_chat, outputs=history_box)

demo.launch()
























# import gradio as gr
# from connect_with_llm import load_resources, run_conversational_rag_pipeline

# retriever, chain = load_resources()

# # Global chat history and last_context for follow-ups
# chat_history = []
# last_context = ""

# # Enhanced respond to support follow-ups
# def respond(user_input, history):
#     global chat_history, last_context

#     # Check if input is a new laptop query (simple keyword check)
#     laptop_keywords = ["laptop", "notebook", "macbook", "chromebook", "gaming", "ultrabook"]
#     is_new_query = any(word in user_input.lower() for word in laptop_keywords)

#     if is_new_query:
#         # Run the full RAG pipeline to get recommendations
#         response = run_conversational_rag_pipeline(user_input, retriever, chain)
#         # Save context for follow-ups from last successful response
#         # (You need to modify run_conversational_rag_pipeline to also return context OR store context globally)
#         # For now, just append response to chat_history
#         last_context = response  # This is just a placeholder; ideally you'd save the raw retrieved context
#     else:
#         # If follow-up, ideally call a different function that uses last_context and follow-up query
#         if not last_context:
#             response = "⚠️ Please ask a laptop-related question first."
#         else:
#             # You would add your follow-up handling here, e.g. call a followup_chain
#             response = "Follow-up handling not yet implemented. Please ask a main query."

#     chat_history.append((user_input, response))
#     return chat_history, chat_history

# with gr.Blocks() as demo:
#     gr.Markdown("# 💻 AI Laptop Assistant")
#     gr.Markdown("Ask about laptops with price, RAM, or usage needs!")

#     chatbot = gr.Chatbot()
#     msg = gr.Textbox(placeholder="Ask me something like: best laptops under ₹60000 with 8GB RAM", label="Your query")

#     clear = gr.Button("Clear chat")

#     def clear_chat():
#         global chat_history, last_context
#         chat_history = []
#         last_context = ""
#         return [], ""

#     msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
#     clear.click(clear_chat, outputs=[chatbot, msg])

# demo.launch()


