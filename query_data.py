import sys
from dataclasses import dataclass
from secret_key import openai_key
import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

PROMPT_TEXT = """
Please answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
CHROMA_DB_PATH = "chroma"
os.environ['OPENAI_API_KEY'] = openai_key

def main():
    if len(sys.argv) != 2:
        print("Need to provide one input query in command line prompt as a string with quotation marks.")
        return
    # get chroma database of vectors
    openai_embedding_function = OpenAIEmbeddings()
    chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=openai_embedding_function)
    query = sys.argv[1]
    # perform vector search
    results = chroma_db.similarity_search_with_relevance_scores(query, k = 3)
    if len(results) == 0 or results[0][1] < .65:
        print("Unable to find accurate results")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEXT)
    context = "\n\n - - - \n\n".join([doc.page_content for doc, _ in results])
    prompt = prompt_template.format(context=context, question=query)
    chat_model = ChatOpenAI()
    sources = [doc.metadata for doc, _ in results]
    response = chat_model.predict(prompt)
    print("Response ", response + "\n")
    print("Source ", sources)
    
if __name__ == "__main__":
    main()