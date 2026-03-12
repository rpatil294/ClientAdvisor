import os
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

loader = {
    "*.txt": TextLoader,
    "*.pdf": PyPDFLoader,
    "*.csv": CSVLoader,
    "*.md": UnstructuredMarkdownLoader,
}

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_store")
print(f"Loading documents from: {KNOWLEDGE_DIR}")

all_docs = []
for glob_pattern, loader_cls in loader.items():
    dir_loader = DirectoryLoader(KNOWLEDGE_DIR,
                             glob=glob_pattern,
                             loader_cls=loader_cls)
    all_docs.extend(dir_loader.load())
print(f"Loaded {len(all_docs)} documents.")

if not all_docs:
    print("No documents found in the knowledge store")

text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=600,
      chunk_overlap=100,
      add_start_index=True,
      strip_whitespace=True,
)

docs_processed = text_splitter.split_documents(all_docs)
print(f"Split into {len(docs_processed)} chunks after processing.") 

embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

vectorstore = Chroma(
    collection_name="knowledge_store",
    embedding_function=embedding_model,
)

if docs_processed:
    vectorstore.add_documents(documents=docs_processed)
    print(f"Added {len(docs_processed)} documents to the vector store.")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2, 
        "fetch_k": 5, 
        "lambda_mult": 0.5
    },
)


@tool("knowledge_search", return_direct=False)
def knowledge_search(query: str) -> str:
    """Search the knowledge_store for relevant information about strategies and financial insights to inform the analyst's research. Use this tool to retrieve information that can help the analyst provide better recommendations to the client but not to answer client's questions on live data or market prices"""
    results = retriever.invoke(query)

    if not results:
        return "No relevant documents found in the knowledge store."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] (source: {source})\n{doc.page_content}")
    print(f"Knowledge Search Results for query: '{query}'\n" + "\n\n".join(formatted))
    return "\n\n".join(formatted)
