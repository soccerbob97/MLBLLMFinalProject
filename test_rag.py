import sys
from dataclasses import dataclass
from secret_key import openai_key
import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import time
PROMPT_TEXT = """
Please answer the question based only on the following context:

{context}

---

Answer the question based on the above context: 

{question}

A. {A}
B. {B}
C. {C}
D. {D}

Only write one letter for the answer: A, B, C, or D
"""

CHROMA_DB_PATH = "biology_lecture_notes_chroma2"
os.environ['OPENAI_API_KEY'] = openai_key

def main():
    #if len(sys.argv) != 2:
    #    print("Need to provide one input query in command line prompt as a string with quotation marks.")
    #    return
    # get chroma database of vectors
    start_time = time.time()
    openai_embedding_function = OpenAIEmbeddings()
    chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=openai_embedding_function)
    df = pd.read_csv('college_biology_test.csv')
    accuracy = 0
    for index in df.index:
        query = df['Question'][index]
        choice_a = df['A'][index]
        choice_b = df['B'][index]
        choice_c = df['C'][index]
        choice_d = df['D'][index]
        # perform vector search
        results = chroma_db.similarity_search_with_relevance_scores(query, k = 3)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEXT)
        context = "\n\n - - - \n\n".join([doc.page_content for doc, _ in results])
        prompt = prompt_template.format(context=context, question=query, A=choice_a, B=choice_b, C=choice_c, D=choice_d)
        #print("prompt ", prompt)
        chat_model = ChatOpenAI()
        sources = [doc.metadata for doc, _ in results]
        response = chat_model.predict(prompt)
        letter_response = response[0]
        print("Response ", response)
        print("first letter ", response[0])
        print("Source ", sources)
        label = df['Label'][index]
        print("label ", label)
        if letter_response == label:
            accuracy += 1
            print("accuracy increase ", accuracy)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    print("accuracy ", accuracy/df.shape[0])
    #print("no results ", no_results/df.shape[0])

main()