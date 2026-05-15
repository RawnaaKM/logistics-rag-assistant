import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}
"""

def query_rag(query_text: str, top_k: int,db):
    
    results = db.similarity_search_with_score(query_text, k=top_k)
    
    sources = [f"{doc.metadata.get('source')} | Page {doc.metadata.get('page')}"for doc, score in results]
    
    retrieved_chunks = [doc.page_content for doc, score in results]
    
    context_text = "\n\n---\n\n".join(retrieved_chunks)    
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text
    )
    
    model = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key="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"
    )
    
    response = model.invoke(prompt).content
    
    return response, retrieved_chunks, sources