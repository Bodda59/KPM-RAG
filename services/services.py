from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from services import db_service
from services import lesson_service

class PDFToChromaETL:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir

    def extract(self, pdf_path):                 # E
        return PyPDFLoader(pdf_path).load()

    def transform(self, docs):                   # T
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 150)
        return splitter.split_documents(docs)
        

         
    def load(self, chunks):                      # L
        return db_service.add_docs(chunks)

    def run(self, pdf_path):
        docs = self.extract(pdf_path)
        chunks = self.transform(docs)
        db = self.load(chunks) 
        print(f"Loaded {len(chunks)} chunks")
        return db

if __name__ == "__main__":
    etl = PDFToChromaETL()
    db = etl.run("./data/langchain_demo.pdf")

