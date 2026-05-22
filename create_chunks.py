# Imports

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import json

# Paths and Model

CHROMA_PATH = r"C:\Users\Clayton Miranda\Desktop\Modelos\Testes\Clay Bank Assistant (Assistant Agent + RAG)\chroma"

DATA_PATH = r"C:\Users\Clayton Miranda\Desktop\Modelos\Testes\Clay Bank Assistant (Assistant Agent + RAG)\clay_bank_documentation"

JSON_PATH = r"C:\Users\Clayton Miranda\Desktop\Modelos\Testes\Clay Bank Assistant (Assistant Agent + RAG)\chunks_json\chunks.json"

OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Main

def main():

    generate_data_store()

# Gerate Data Store

def generate_data_store():

    documents = load_documents()

    chunks = split_text(documents)

    export_chunks_to_json(chunks)

    save_to_chroma(chunks)

# Load documents from PDF directory

def load_documents():

    loader = PyPDFDirectoryLoader(DATA_PATH)

    documents = loader.load()

    print(f"\nTotal de páginas carregadas: {len(documents)}")

    # Exemplo
    if len(documents) > 0:

        print("\n===== EXEMPLO DE PÁGINA =====")

        print(documents[0].metadata)

    return documents

# Split text into chunks

def split_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nSplit {len(documents)} páginas em {len(chunks)} chunks.")

    # adding custom metadata to chunks

    for i, chunk in enumerate(chunks):

        chunk_id = f"chunk_{i}"

        page = chunk.metadata.get("page", 0) + 1

        source = os.path.basename(
            chunk.metadata.get("source", "unknown")
        )

        start_index = chunk.metadata.get("start_index", 0)

        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["page"] = page
        chunk.metadata["source"] = source
        chunk.metadata["start_index"] = start_index

    # Chunk example
    if len(chunks) > 0:

        print("\n===== EXEMPLO DE CHUNK =====")

        print("\nCONTEÚDO:\n")

        print(chunks[0].page_content)

        print("\nMETADATA:\n")

        print(chunks[0].metadata)

    return chunks

# Export chunks to JSON

def export_chunks_to_json(chunks: list[Document]):

    # Create directory
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    json_data = []

    for chunk in chunks:

        item = {

            "chunk_id": chunk.metadata.get("chunk_id"),

            "page": chunk.metadata.get("page"),

            "source": chunk.metadata.get("source"),

            "start_index": chunk.metadata.get("start_index"),

            "content": chunk.page_content
        }

        json_data.append(item)

    with open(JSON_PATH, "w", encoding="utf-8") as f:

        json.dump(
            json_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nJSON salvo em:\n{JSON_PATH}")

# Salving chunks to Chroma vector store

def save_to_chroma(chunks: list[Document]):

    # Overwrite existing Chroma database
    if os.path.exists(CHROMA_PATH):

        shutil.rmtree(CHROMA_PATH)

    # Ollama embeddings
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL
    )

    # Create Chroma vector store
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    db.persist()

    print(f"\nSaved {len(chunks)} chunks to:")
    print(CHROMA_PATH)

# Execution

if __name__ == "__main__":

    main()