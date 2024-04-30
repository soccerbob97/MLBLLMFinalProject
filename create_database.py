from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from secret_key import openai_key
#import unstructured
import os
import shutil
import sys
print(sys.path)

os.environ['OPENAI_API_KEY'] = openai_key
CHROMA_DB_PATH = "biology_lecture_notes_chroma3"
FILE_PATH = "./LectureNotes"

def generate_database():
    # load documents 
    loader = DirectoryLoader(FILE_PATH,glob="*.pdf")
    documents = loader.load()
    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_DB_PATH}")

if __name__ == "__main__":
    generate_database()
    
