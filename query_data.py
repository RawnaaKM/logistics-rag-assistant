import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
You are a logistics question-answering assistant.

Use the provided context to answer the question accurately.
You must answer the question based only on the following context:

{context}

---

Question: {question}
"""

def query_rag(query_text: str, top_k: int, db):

    results = db.similarity_search_with_score(query_text, k=top_k)

    sources = [
        f"{doc.metadata.get('source')} | Page {doc.metadata.get('page')}"
        for doc, score in results
    ]

    retrieved_chunks = [
        doc.page_content for doc, score in results
    ]

    context_text = "\n\n---\n\n".join(retrieved_chunks)

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    ).format(
        context=context_text,
        question=query_text
    )

    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        model = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0
        )

        response = model.invoke(prompt)

        return response.content, retrieved_chunks, sources

    except Exception as e:
        return f"ERROR: {str(e)}", retrieved_chunks, sources
