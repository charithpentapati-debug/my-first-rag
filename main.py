import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings()

# Ask for PDF first
while True:
    pdf_path = input("Enter PDF filename: ")
    if os.path.exists(pdf_path):
        break
    print(f"File '{pdf_path}' not found. Try again.")

# Create dynamic index name
index_name = os.path.splitext(pdf_path)[0] + "_index"

# Build or load vector store
if not os.path.exists(index_name):
    print("Building vector store...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_name)

    print("Vector store saved.")

else:
    print("Loading existing vector store...")

    vectorstore = FAISS.load_local(
        index_name,
        embeddings,
        allow_dangerous_deserialization=True
    )

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    result = chain.invoke(query)

    print(result.content)