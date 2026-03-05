# LLM Hallucination Detection & Correction Using RAG

A Retrieval-Augmented Generation (RAG) assistant that reduces hallucinations by grounding answers in real-time web content. The app takes a user question, searches relevant pages, scrapes them, stores chunks in Pinecone, and generates a context-grounded answer.

## How It Works

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Ask a Question  │ ──► │ Google Search      │ ──► │ Scrape Top URLs  │
│ (chat input)    │     │ (SerpAPI, top 5)   │     │ (WebBaseLoader)  │
└─────────────────┘     └────────────────────┘     └────────┬─────────┘
                                                             │
                                                             ▼
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Pinecone Store   │ ◄── │ Chunk + Embed      │ ◄── │ Split Documents  │
│ (rag-embedds)    │     │ (nomic-embed-text) │     │ (2000 / 100)     │
└────────┬─────────┘     └────────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│ Retrieve Top 3   │ ──► │ Build Prompt       │ ──► │ Generate Answer  │
│ similar chunks   │     │ with context        │     │ (llama3.2)       │
└──────────────────┘     └────────────────────┘     └──────────────────┘
```

### Step-by-Step Process

1. User enters a question in Streamlit chat.
2. App fetches related Google results using SerpAPI and keeps top 5 links.
3. App fetches related Google Images (top 4) and displays them.
4. App scrapes content from discovered URLs with WebBaseLoader.
5. Documents are split into chunks (chunk size 2000, overlap 100).
6. Chunks are embedded with nomic-embed-text (768 dimensions).
7. Embedded chunks are inserted into Pinecone index rag-embedds.
8. Similarity search retrieves top 3 relevant chunks.
9. llama3.2 generates a detailed response using retrieved context.

## Features

- Question-first workflow (no manual URL entry required)
- Web search via SerpAPI to discover fresh sources
- Related image preview for user context
- Pinecone-backed persistent vector storage
- Ollama local models for embeddings and answer generation
- Streamlit UI with cached model loading

## Prerequisites

- Python 3.10+
- Ollama installed and running
- Pinecone account
- SerpAPI account and API key

## Local Setup

### 1) Clone and enter project

```bash
git clone https://github.com/Rishiraj-Pathak-27/LLM-Hallucination-Detection-Correction-Using-RAG.git
cd LLM-Hallucination-Detection-Correction-Using-RAG
```

### 2) Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Install and start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

### 5) Create Pinecone index

Create an index with:

- Index name: rag-embedds
- Dimension: 768
- Metric: cosine

Important: The embedding model nomic-embed-text outputs 768-d vectors. If index dimension is different, upsert fails.

### 6) Configure API keys

Current code reads keys from constants in rag_scrapper.py and sets environment variables at runtime.

You should replace hardcoded values with your own keys or move fully to environment variables:

```bash
export SERPAPI_API_KEY="your-serpapi-key"
export PINECONE_API_KEY="your-pinecone-key"
```

### 7) Run app

```bash
streamlit run rag_scrapper.py
```

## Usage

1. Ask any question in the chat box.
2. App shows related images and discovered source URLs.
3. App scrapes, chunks, embeds, and stores content in Pinecone.
4. App retrieves relevant chunks and returns a grounded answer.

## Project Structure

```
├── rag_scrapper.py
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Streamlit | Interactive app |
| Search | SerpAPI | Web links + image results |
| Scraping | WebBaseLoader | Load web page content |
| Splitter | RecursiveCharacterTextSplitter | Chunk long text |
| Embeddings | nomic-embed-text (Ollama) | Vectorize text |
| Vector DB | Pinecone | Store/retrieve embeddings |
| LLM | llama3.2 (Ollama) | Context-grounded answer |
| Orchestration | LangChain | Pipeline composition |

## Troubleshooting

| Issue | Fix |
|------|-----|
| Vector dimension mismatch | Recreate Pinecone index with dimension 768 |
| Pinecone API key missing | Set PINECONE_API_KEY correctly |
| SerpAPI errors | Check SERPAPI_API_KEY and account quota |
| No scraped docs | Query may return blocked/unreachable URLs |
| model not found | Run ollama pull llama3.2 and ollama pull nomic-embed-text |

## Security Note

Do not commit real API keys to source control. Move keys to environment variables before publishing this project.

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, open an issue first.
