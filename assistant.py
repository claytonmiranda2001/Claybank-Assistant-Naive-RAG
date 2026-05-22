# Imports

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Assistant Agent

class ClayBankAssistant:

    CHROMA_PATH = r"C:\\Users\\Clayton Miranda\\Desktop\\Modelos\\Testes\\Clay Bank Assistant (Assistant Agent + RAG)\\chroma"

    PROMPT_TEMPLATE = """
    You are ClayBank Assistant.

    Your job is to answer customer questions using ONLY the provided context.

    Rules:
    - Be professional
    - Be concise
    - Never invent information
    - If the answer is not in the context, say:
      "I could not find this information in the available documentation."
    - Mention source and page when available

    Context:
    {context}

    Question:
    {question}
    """
    #Models and tools
    def __init__(self):
        # Embedding model
        self.embedding_function = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        self.db = Chroma(
            persist_directory=self.CHROMA_PATH,
            embedding_function=self.embedding_function
        )
        # Answer generation model
        self.model = ChatOllama(
            model="llama3"
        )
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            self.PROMPT_TEMPLATE
        )
    
    # Query, context retrieval and response generation
    def ask(self, question):

        # Vector search
        results = self.db.similarity_search_with_score(
            question,
            k=3
        )

        # Context
        context_parts = []

        for doc, score in results:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "Unknown")

            context_parts.append(
                f'''
SOURCE: {source}
PAGE: {page}
CHUNK_ID: {chunk_id}
SCORE: {score}

CONTENT:
{doc.page_content}
'''
            )

        context_text = "\n\n---\n\n".join(context_parts)

        # Prompt
        prompt = self.prompt_template.format(
            context=context_text,
            question=question
        )

        # Response generation
        response = self.model.invoke(prompt)

        return response.content