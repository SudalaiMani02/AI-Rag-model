import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Lazy loading variables
embedding_model = None
index = None
chunks = None
metadata = None
total_pages = None


def load_rag_system():
    global embedding_model, index, chunks, metadata, total_pages

    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if index is None:
        print("Loading FAISS index...")
        index = faiss.read_index("vector.index")

    if chunks is None:
        print("Loading chunk data...")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        metadata = data["metadata"]
        total_pages = data["total_pages"]

        print("RAG system loaded successfully")


def ask_question(question):

    # Load system only when first request comes
    load_rag_system()

    q = question.lower().strip()

    greetings = ["hi", "hello", "hey"]
    words = q.split()

    if any(word in greetings for word in words):
        return "Hello! How may I assist you today?"

    thanks = ["thanks", "thank you"]

    if any(word in q for word in thanks):
        return "You're welcome. Is there anything else I can assist you with?"

    exit_words = ["bye", "exit", "quit"]

    if any(q.startswith(word) for word in exit_words):
        return "Alright! If you need help later, feel free to ask."

    if not q:
        return "Please enter a valid question."

    try:

        query_vector = embedding_model.encode([question])
        query_vector = np.array(query_vector).astype("float32")

        scores, indices = index.search(query_vector, 5)

        context_parts = []

        for score, idx in zip(scores[0], indices[0]):
            chunk_text = chunks[idx]
            context_parts.append(chunk_text)

        context = "\n\n".join(context_parts)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"""
        You are KADIT AI Assistant, designed to help users understand the uploaded company policy document.
        Your main objective is to answer questions strictly related to the provided document.
        Instructions:
        1. Always check the provided context first before answering.
        2. If the user's question is related to the document, answer only using the information from the context.
        3. If the question is partially related, answer using only the relevant information from the context.
        4. If the question is NOT related to the document, do NOT generate an answer.
        
        Instead, politely respond with:
        - Respond politely by saying you do not have information about this in the available company documents.
        - Suggest contacting HR or referring to the official company policy portal.
        5. If the user asks about the assistant's previous response or system behavior, respond politely explaining that the relevant section may not have been retrieved earlier.
        Do not say the information is unavailable in the document for such questions.
        6. If the user greets you or starts the conversation, respond politely like:
        "Hello! I am KADIT AI Assistant. How may I assist you regarding our policy document?"
        7.Read the provided context carefully and summarize the relevant information clearly instead of copying sentences directly from the context.
        Example response when information is not available:
        "I couldn't find information about this topic in the available company documents. For accurate guidance, please refer to the official policy portal or contact HR.."
        Rules:
        - Do not create information that is not present in the document.
        - Do not answer general questions unrelated to the policy document.
        - Do not generate emails, suggestions, or external advice.
        - Keep answers concise and professional.
        The document contains {total_pages} pages."""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong while processing your question."
