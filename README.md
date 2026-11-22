# 📚 AI Document Assistant — Hybrid RAG

Ever wanted to ask questions about a PDF and get answers based on its actual content instead of searching through pages manually?

That's what I built with this project!

The **AI Document Assistant** is a PDF-based question-answering application that uses Retrieval-Augmented Generation (RAG). You can upload your knowledge through a prepared PDF document, ask questions, and receive answers generated using the retrieved information.

I built this project to understand how a RAG pipeline works in practice, from extracting PDF text to retrieving relevant information and generating answers with an LLM.

## ✨ Features

- **PDF Text Extraction:** Extracts text from PDF documents using PyPDF.
- **Text Chunking:** Splits extracted text into smaller, manageable chunks with overlap.
- **Semantic Search:** Uses Sentence Transformers to convert text into embeddings and FAISS to find semantically relevant chunks.
- **Keyword Search:** Uses BM25 to retrieve chunks based on keyword relevance.
- **Hybrid Search:** Combines semantic and keyword search results using Reciprocal Rank Fusion (RRF).
- **Answer Generation:** Uses Google's Gemini API to generate answers based on the retrieved context.
- **Retrieval Evaluation:** Includes Precision@K, Recall@K, Hit Rate@K, and Mean Reciprocal Rank (MRR) to evaluate retrieval quality.
- **Streamlit Interface:** Provides a simple interface to ask questions and view generated answers and retrieved chunks.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| PDF Processing | PyPDF |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Hybrid Ranking | Reciprocal Rank Fusion |
| LLM | Google Gemini API |
| Frontend | Streamlit |
| Environment Management | uv |

## ⚙️ How It Works

The application follows a Hybrid RAG pipeline:

1. **Document Loading:** Reads the PDF and extracts its text.
2. **Chunking:** Divides the extracted text into smaller chunks with overlapping content.
3. **Embedding Generation:** Converts each chunk into a numerical vector using the `all-MiniLM-L6-v2` embedding model.
4. **Indexing:** Stores the embeddings in a FAISS index and prepares a BM25 index for keyword retrieval.
5. **Hybrid Retrieval:** Searches for relevant chunks using both FAISS and BM25.
6. **Result Fusion:** Combines the rankings using Reciprocal Rank Fusion (RRF).
7. **Answer Generation:** Sends the retrieved context and user's question to Gemini to generate an answer.

The goal is to provide answers grounded in the document rather than relying only on the LLM's general knowledge.

## 📁 Project Structure

```text
ai-document-assistant/
│
├── data/
│   └── AI_ML_Fundamentals_RAG_Dataset.pdf
│
├── app.py
├── main.py
├── test_gemini.py
├── pyproject.toml
├── README.md
├── .gitignore
└── .env
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ragavi2102/Ai-document-assistant.git
cd Ai-document-assistant
```

### 2. Create a virtual environment

Install [uv](https://docs.astral.sh/uv/) if you haven't already.

```bash
uv venv --python 3.12
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure the Gemini API key

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own API key.

**Important:** Never share your API key or commit your `.env` file to GitHub.

### 5. Run the application

```bash
uv run --active streamlit run app.py
```

The application will open in your browser.

## 📊 Retrieval Evaluation

I also included a small retrieval evaluation component to understand how effectively the system retrieves relevant information.

The application supports these metrics:

- **Precision@K:** Measures how many retrieved chunks are relevant.
- **Recall@K:** Measures how many of the relevant chunks in the document were retrieved.
- **Hit Rate@K:** Checks whether at least one relevant chunk was retrieved.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant result is ranked.

Relevant chunks are labelled manually to provide reference values for evaluation.

## 🎯 What I Learned

Working on this project helped me understand how individual components of a RAG system work together.

Some key concepts I explored:

- Document processing and text chunking.
- Semantic search versus keyword search.
- Embeddings and vector databases.
- Hybrid retrieval and Reciprocal Rank Fusion.
- Connecting an LLM API to a retrieval pipeline.
- Evaluating retrieval performance.

## 🔮 Future Improvements

- Support multiple PDF documents.
- Improve chunking and retrieval strategies.
- Add source citations to generated answers.
- Explore more advanced retrieval techniques.
- Add automated evaluation using a larger question-answer dataset.

## 👩‍💻 Author

**Ragavi S**

[GitHub](https://github.com/Ragavi2102)

---

*Built as a hands-on project to explore Retrieval-Augmented Generation, hybrid search, and LLM-based applications.*
