import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_ollama import ChatOllama
from services.services import db_service

load_dotenv()
ollama_bearer_token = os.getenv("OLLAMA_API_KEY", "")

model = ChatOllama(
    model="gpt-oss:120b:cloud",
    base_url="https://api.ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": "Bearer " + ollama_bearer_token
        }
    },
)

QA_TEMPLATE = """
Use the following context and conversation history to answer the question.
If you don't find the answer in the context, don't try to make up an answer.
If you find the answer, use all relevant context to make the answer.

Context:
{context}

Conversation so far:
{history}

Question: {question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

llm_chain = LLMChain(
    llm=model,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=True,
)

document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="Context:\ncontent: {page_content}",
)

combine_document_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)


def get_response(query: str, history: str = "") -> str:
    """
    Retrieve relevant chunks from the vector db and generate an answer,
    using prior conversation turns for context.
    """
    retriever = db_service.vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    if not docs:
        return "I don't know — no relevant information was found."

    result = combine_document_chain.invoke({
        "input_documents": docs,
        "question": query,
        "history": history,
    })

    return result["output_text"]