# My First RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you query any PDF document using natural language.

## What it does
- Accepts any PDF — prompts user for filename with input validation
- Each PDF gets its own saved index (e.g. Think_Straight_index)
- Splits it into chunks and converts them to embeddings
- Stores embeddings in a FAISS vector store
- Retrieves relevant chunks based on your question
- Uses GPT-3.5-turbo to generate answers from the retrieved context

## Tech Stack
- Python
- LangChain
- OpenAI API (GPT-3.5-turbo + text-embedding-ada-002)
- FAISS
- PyPDF

## How to run

1. Clone the repo
2. Install dependencies:

pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf python-dotenv

3. Create a `.env` file with your OpenAI API key:

OPENAI_API_KEY=your-key-here

4. Add your PDF to the project folder and update the filename in `main.py`
5. Run:

python main.py

## Notes
- Each PDF generates its own named index — no re-embedding on repeat queries for the same document
- Answers are generated strictly from the document context, not from general LLM knowledge
