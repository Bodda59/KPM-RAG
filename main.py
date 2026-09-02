from services import db_service
from services import lesson_service
from typing import Optional
from enum import Enum
from pydantic import BaseModel
from fastapi import UploadFile, File
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from os import listdir
import shutil
from services.services import PDFToChromaETL

app = FastAPI(title = "KPM-RAG", version = "1.0.0", description = "RAG API")

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://0.0.0.0:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

directory = "./data"

class LessonStatus(str, Enum):
    new = "new"
    update = "update"
    delete = "delete"


class lesson(BaseModel):
    id: Optional[int] = None
    pdf_path: str
    status: LessonStatus

etl = PDFToChromaETL()


@app.post('/upload-file')
def upload_file(uploaded_file: UploadFile = File(..., alias="file")):
    """
    upload new file to files directory.

    Args:
        uploaded_file (file): the pdf input file.

    Returns:
        result (object): the file data and process result.
    """

    """
    Implement your logic here to save the uploaded file to the desired directory. For example, you can use the shutil library to save the file to a specific folder.
    """
    dest_path = os.path.join(directory, uploaded_file.filename)
    with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

    return {
        'uploaded': True,
        'message' : 'File Uploaded successfully'
    }

@app.get('/process-batch-files')
def process_batch_files():
    """
    process, and save a new pdf file into db.

    Returns:
        result (object): the file data and process result.
    """
    
     
    files = listdir(directory)
    if files:
        for fileName in files:
           """
           Implement your logic here to process each file. For example, you can call a function from the files_service to load and process the PDF file.
           """
           
           db = etl.run(os.path.join(directory, fileName))
        return {
        'uploaded': True,
        'message' : 'Files processed successfully'
    }
    else:   
        return {
        'uploaded': False,
        'message' : 'No files to be processed!'
    }




@app.post('/process-lesson')
def process_lesson(lesson: lesson):
    """
    process, and save a new lesson into db, update, delete from db.

    Args:
        lesson (lesson): lesson body.

    Returns:
        state (state): the lesson data and process result.
    """
    match lesson.status:
        case "new":
            """
            Implement your logic here to process a new lesson. For example, you can call a function from the lesson_service to save the new lesson into the database.
            """
            chunks = lesson_service.preprocess_lesson(lesson)
            db_service.add_docs(chunks)
            return {
                'processed': True,
                'message' : 'Lesson processed successfully'
            }
        case "update":
            """
            Implement your logic here to process an update to an existing lesson. For example, you can call a function from the lesson_service to update the lesson in the database.
            """
            chunks = lesson_service.preprocess_lesson(lesson)
            db_service.update_docs(lesson.id,chunks)
            return {
                'processed': True,
                'message' : 'Lesson processed successfully'
            }
        case "delete":
            """
            Implement your logic here to process the deletion of a lesson. For example, you can call a function from the lesson_service to delete the lesson from the database.
            """
            db_service.delete_docs(lesson.id)
            return {
                'processed': True,
                'message' : 'Lesson processed successfully'
            }    
        case default:
            return 0

@app.post('/get-chunks')
def get_chunks(query: str):
    """
    get chunks from db.

    Returns:
        result list(objects): the chunks data.
    """
    """
    Implement your logic here to retrieve chunks from the database based on the provided query. For example, you can call a function from the files_service to fetch the relevant chunks.
    """
    chunks = db_service.vector_store.similarity_search(query,k=3)
    return {
        'chunks': chunks,
        'message' : 'Chunks retrieved successfully'
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)