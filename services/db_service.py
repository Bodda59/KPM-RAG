

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

def add_docs(chunks):
    vector_store.add_documents(chunks, embedding = embedding)
    return True

def delete_docs(lesson_id):
    vector_store.delete(ids=[lesson_id])
    return {
        'updated': True,
    }

def update_docs(lesson_id,docs):
    vector_store.update(ids=[lesson_id],docs=docs)
    return {
        'updated': True,
    }



