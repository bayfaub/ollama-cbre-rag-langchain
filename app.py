from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=False)

db = Chroma(persist_directory='chroma', embedding_function=embeddings)

retriever = db.as_retriever(search_type='similarity')

llm = ChatOllama(model='gemma2', keep_alive="2h")

template = """
You are a question/answer chatbot built for real estate brokers to summarize CBRE research data.\
Answer the users questions based only on the context and extract out a meaningful answer.\
Please write in full sentences with correct spelling and punctuation. If it makes sense you can list out different parts of an answer. \
If you can't find the answer in the context then just say you are unable to determine the answer.\
If you are requested to do math for the user then you can do that. \
If the users asks you for specific investing advice, please say that you don't have enough data to make intelligent investment decisions.

CONTEXT: {context}

QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)

def ask_question(question: str):
    print("Answer(please wait while I think):\n")
    for chunk in chain.stream(question):
        print( chunk.content , end="", flush=True)
    print('\n')
    

if __name__ == "__main__":
    while True:
        question = input("Please input a question relating to commercial real estate (or type 'quit' to exit):\n")
        if question.lower() == 'quit':
            break
        answer = ask_question(question)
        print('Answer complete.')