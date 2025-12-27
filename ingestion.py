import asyncio
import os
import ssl
from  typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_ollama import OllamaEmbeddings
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings( model="nomic-embed-text")

vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
taviliy_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()



async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_header("Vector Stotage Phase")
    log_info(f"VectorStore Indexing: Prepare to add {len(documents)} documents to vector store", Colors.DARKCYAN)

    batches = [documents[i : i+ batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")


    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorStore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)")

        except Exception as e:
            log_error(f"VectorStore indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully")


async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyCrawl: Starting to crawl from https://python.langchain.com/", 
              Colors.PURPLE)
    
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
    })

    all_docs = [Document(page_content=result['raw_content'], 
                         metadata={"source": result['url']}) 
                         for result in res["results"]] 
    
    log_success(f"TavilyCrawl: Successfully crawl {len(all_docs)} URL from documentation site")
    
    log_header("Document Chunking Phase")
    log_info(
        "Text splitter processing", Colors.YELLOW
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(f"Text splitter: Created {len(splitted_docs)}  chunks from {len(all_docs)} documents")

    await index_documents_async(splitted_docs, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())



