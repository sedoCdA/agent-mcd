import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

SOLUTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "solutions.txt")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

_vector_store = None


def build_vector_store():
    """
    Loads the solution document, splits it into chunks,
    embeds them, and saves the FAISS index to disk.
    """
    print("Building vector store from solution documents...")

    loader = TextLoader(SOLUTIONS_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Loaded docs: {len(documents)}")
    print(f"Document length: {len(documents[0].page_content)}")
    print(f"Generated chunks: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No chunks generated from documents")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(VECTOR_STORE_PATH)

    print(f"Vector store saved. Total chunks: {len(chunks)}")

    return vector_store

def load_vector_store():
    """
    Loads the FAISS index from disk if it exists,
    otherwise builds it first.
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_STORE_PATH):
        _vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        _vector_store = build_vector_store()

    return _vector_store


def retrieve_solution(query: str, top_k: int = 2) -> str:
    """
    Retrieves the most relevant solution chunks for a given query.
    Returns them as a single context string.
    """
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=top_k)

    if not results:
        return "No relevant solution found in the knowledge base."

    context = "\n\n".join([doc.page_content for doc in results])
    return context


if __name__ == "__main__":
    build_vector_store()
    test_query = "inventory is not syncing"
    context = retrieve_solution(test_query)
    print(f"Query: {test_query}")
    print(f"Retrieved context:\n{context}")