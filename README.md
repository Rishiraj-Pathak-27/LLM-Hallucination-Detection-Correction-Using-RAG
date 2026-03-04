# LLM Hallucination Detection & Correction Using RAG

A Retrieval-Augmented Generation (RAG) web scraper that helps reduce LLM hallucinations by grounding responses in real web content. This application scrapes web pages, stores them in a vector database, and uses retrieved context to provide accurate, fact-based answers.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────┐
│  Enter URL  │ ──► │ Scrape Page  │ ──► │ Split into      │ ──► │ Generate  │
│             │     │ (WebLoader)  │     │ Chunks (2000)   │     │ Embeddings│
└─────────────┘     └──────────────┘     └─────────────────┘     └─────┬─────┘
                                                                       │
                                                                       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────┐
│   Answer    │ ◄── │  LLM (Llama  │ ◄── │ Retrieve Top 3  │ ◄── │  Pinecone │
│  (Grounded) │     │    3.2)      │     │ Similar Chunks  │     │  Vector DB│
└─────────────┘     └──────────────┘     └─────────────────┘     └───────────┘
```

### Step-by-Step Process

1. **URL Input**: User enters a webpage URL to index
2. **Web Scraping**: `WebBaseLoader` fetches the page content
3. **Text Splitting**: Content is split into 2000-character chunks with 100-char overlap
4. **Embedding Generation**: `nomic-embed-text` model creates 768-dimensional embeddings for each chunk
5. **Vector Storage**: Embeddings are stored in Pinecone vector database
6. **Question Input**: User asks a question about the indexed content
7. **Similarity Search**: Top 3 most relevant chunks are retrieved from Pinecone
8. **Answer Generation**: `llama3.2` generates an answer using only the retrieved context
9. **Hallucination Reduction**: By grounding responses in actual web content, hallucinations are minimized

## Features

- **Fast Embeddings**: Uses `nomic-embed-text` (768 dims) - ~50x faster than LLM-based embeddings
- **Accurate Answers**: `llama3.2` for high-quality response generation
- **Persistent Storage**: Pinecone vector database for scalable, persistent storage
- **Streamlit UI**: Clean, interactive web interface
- **Caching**: Model caching for faster subsequent requests

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- [Pinecone](https://www.pinecone.io/) account (free tier available)

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Rishiraj-Pathak-27/LLM-Hallucination-Detection-Correction-Using-RAG.git
cd LLM-Hallucination-Detection-Correction-Using-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Start Ollama server (if not running)
ollama serve
```

### 5. Set Up Pinecone

1. Create a free account at [Pinecone](https://www.pinecone.io/)
2. Create a new index with:
   - **Index Name**: `rag-index`
   - **Dimensions**: `768` (matches nomic-embed-text)
   - **Metric**: `cosine`
3. Copy your API key

### 6. Configure API Key

Open `rag_scrapper.py` and replace the placeholder with your Pinecone API key:

```python
api_key = "YOUR_PINECONE_API_KEY_HERE"
```

> ⚠️ **Security Note**: For production, use environment variables instead:
> ```bash
> export PINECONE_API_KEY="your-api-key"
> ```

### 7. Run the Application

```bash
streamlit run rag_scrapper.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Index a Webpage**:
   - Enter a URL in the text field
   - Click "Load & Index"
   - Wait for the success message

2. **Ask Questions**:
   - Type your question in the chat input at the bottom
   - The AI will respond using only the indexed content
   - Answers are limited to 3 sentences for conciseness

3. **Index More Pages**:
   - You can index multiple URLs
   - All content is stored in Pinecone and persists across sessions

## Project Structure

```
├── rag_scrapper.py      # Main application file
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI | Streamlit | Web interface |
| Web Scraping | LangChain WebBaseLoader | Fetch page content |
| Text Splitting | RecursiveCharacterTextSplitter | Chunk documents |
| Embeddings | nomic-embed-text (Ollama) | Convert text to vectors |
| Vector DB | Pinecone | Store and retrieve embeddings |
| LLM | llama3.2 (Ollama) | Generate answers |
| Framework | LangChain | Orchestration |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ollama: command not found` | Install Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` |
| `model not found` | Pull models: `ollama pull llama3.2 && ollama pull nomic-embed-text` |
| `Pinecone connection error` | Check API key and index name |
| `Slow indexing` | Normal for first run; models are cached after |

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
