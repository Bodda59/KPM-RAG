from models import lesson
from langchain.docstore.document import Document

def preprocess_lesson(lesson:lesson):
    docs = []
    id = lesson.id
    lesson_data = get_lesson_data(lesson)
    doc = Document(page_content=lesson_data, id=id)
    docs.append(doc)
    return docs


def get_lesson_data(lesson:lesson):
    del lesson.id
    del lesson.status

    content = lesson.__str__()
    return content
