from dotenv import load_dotenv
load_dotenv()

import os
from typing import List
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings



embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the following question based solely on the provided context."),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        """context:
{context}

question:
{question}

Provide a detailed answer:"""
    )
])



def create_retrieval_chain_with_lcel():
    retrieve_context = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    )

    generate_answer = {
        "question": itemgetter("question"),
        "context": itemgetter("context") | RunnableLambda(format_docs),
        "history": itemgetter("history"),
    } | chat_prompt_template | llm | StrOutputParser()

    return retrieve_context.assign(answer=generate_answer)



def run_llm_chain(question: str, chat_history: List[BaseMessage]) -> dict:
    chain = create_retrieval_chain_with_lcel()

    result = chain.invoke(
        {
            "question": question,
            "history": chat_history[-6:],
        }
    )

    return {
        "answer": result["answer"],
        "context": result["context"],  
    }

