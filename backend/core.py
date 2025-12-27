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


# -----------------------------
# Vector store + embeddings
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -----------------------------
# LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# -----------------------------
# Helpers
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# Prompt
# -----------------------------
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


# -----------------------------
# Chain
# -----------------------------
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


# -----------------------------
# Public API
# -----------------------------
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


    # def normalize_answer(answer):
#     # Gemini / agent responses are often lists of blocks
#     if isinstance(answer, list):
#         return "".join(
#             block.get("text", "")
#             for block in answer
#             if isinstance(block, dict) and block.get("type") == "text"
#         )
#     return str(answer)


# @tool(response_format="content_and_artifact")  # Return two things: 1.serialized 2.retrived_docs
# def retrieve_context(query: str):
#     """Retrieve relevant documentation to help answer user queries about LangChain."""
#     retrieved_docs = vectorStore.as_retriever().invoke(query, k=4)

    # serialized = "\n\n".join(
    #     (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}") for doc in retrieved_docs
    #     )
    
#     return serialized, retrieved_docs



    # """
    # Run the RAG pipeline to answer a query using retrieved documentation.
    
    # Args:
    #     query: The user's question
        
    # Returns:
    #     Dictionary containing:
    #         - answer: The generated answer
    #         - context: List of retrieved documents
    # """

    # system_prompt = (
    #     "You are a helpful AI assistant that answers questions about LangChain documentation. "
    #     "You have access to a tool that retrieves relevant documentation. "
    #     "Use the tool to find relevant information before answering questions. "
    #     "Always cite the sources you use in your answers. "
    #     "If you cannot find the answer in the retrieved documentation, say so." 
    # )

    # rephrase_prompt = 

    # agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # messages = [{"role": "user", "content": query}]

    # response = agent.invoke({"messages": messages})

    # raw_answer = response["messages"][-1].content
    # answer = normalize_answer(raw_answer)

    # context_docs = []
    # for message in response["messages"]:
    #     if isinstance(message, ToolMessage) and hasattr(message, "artifact"):

    #         if isinstance(message.artifact, list):
    #             context_docs.extend(message.artifact)
    
    # return {
    #     "answer": answer,
    #     "context": context_docs
    # }

