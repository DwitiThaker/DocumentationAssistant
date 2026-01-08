# DocumentationAssistant
A **Retrieval-Augmented Generation (RAG)** powered documentation assistant that enables users to query large document collections using natural language. The system combines **LLMs, vector search, and web retrieval** to deliver grounded, context-aware answers through a Streamlit interface.

* * * * *

🚀 Features
-----------

-   Semantic search over documents using vector embeddings

-   Pinecone-backed vector database for fast retrieval

-   Multi-source retrieval (documents + web via Tavily)

-   Google Gemini / Ollama LLM integration

-   Streamlit-based interactive UI

-   LangSmith tracing support for observability

* * * * *

🧠 Architecture Overview
------------------------

```
User Query
   ↓
Streamlit UI
   ↓
Retriever (Pinecone + Tavily)
   ↓
LLM (Gemini / Ollama)
   ↓
Context-aware Answer

```

* * * * *

🛠️ Tech Stack
--------------

-   **Frontend**: Streamlit

-   **LLMs**: Google Gemini, Ollama

-   **Framework**: LangChain

-   **Vector DB**: Pinecone

-   **Web Search**: Tavily

-   **Observability**: LangSmith

* * * * *

📂 Project Structure
--------------------

```
documentationassistant/
│
├── main.py                 # Streamlit entry point
├── backend/
│   ├── core.py             # LLM + RAG pipeline
│   ├── ingestion.py        # Document ingestion & indexing
│   └── logger.py           # Logging utilities
│
├── requirements.txt
└── README.md

```

* * * * *

🔑 Environment Variables
------------------------

### Local Development (`.env`)

Create a `.env` file (do **not** commit it):

```
GOOGLE_API_KEY=your_key
TAVILY_API_KEY=your_key
PINECONE_API_KEY=your_key
INDEX_NAME=your_pinecone_index

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=New_Begin

```

Make sure you load it in code:

```
from dotenv import load_dotenv
load_dotenv()

```

* * * * *

### Streamlit Cloud (Production)

Streamlit **does not read `.env` files**.

Add the following under **App → Settings → Secrets** using **TOML format**:

```
GOOGLE_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
PINECONE_API_KEY = "your_key"
INDEX_NAME = "your_pinecone_index"

LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your_key"
LANGSMITH_PROJECT = "New_Begin"

```

* * * * *

▶️ Running the App Locally
--------------------------

```
pip install -r requirements.txt
streamlit run main.py

```

If you see `inotify instance limit reached`, disable file watching:

```
streamlit run main.py --server.fileWatcherType none

```

* * * * *

⚠️ Common Issues & Fixes
------------------------

### `KeyError: 'INDEX_NAME'`

-   Ensure the variable is set in `.env` (local) or **Streamlit Secrets** (cloud)

### `Invalid format: please enter valid TOML`

-   Secrets must follow `KEY = "value"` format (not `.env` style)

* * * * *

🔒 Security Best Practices
--------------------------

-   Never commit `.env` files or API keys

-   Always rotate exposed keys immediately

-   Use environment validation at startup

-   Prefer read-only Pinecone API keys where possible

* * * * *

🎯 Use Cases
------------

-   Internal documentation Q&A

-   Research paper exploration

-   Knowledge base assistants

-   Developer productivity tools

* * * * *

📌 Future Improvements
----------------------

-   Role-based access control

-   Document upload via UI

-   Streaming responses

-   Caching for repeated queries

* * * * *

👩‍💻 Author
------------

**Dwiti Thaker**\
AI/ML Engineer | RAG Systems | Applied NLP

* * * * *


