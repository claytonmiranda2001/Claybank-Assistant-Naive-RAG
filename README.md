# Claybank-Assistant-Naive-RAG
## Overview

ClayBank AI Assistant is a Retrieval-Augmented Generation (RAG) project designed to simulate a production-ready banking support assistant.

The system uses semantic search and Large Language Models (LLMs) to answer customer questions based exclusively on internal banking documentation.

The project also includes an automated evaluation pipeline capable of validating the quality of generated responses using an LLM-as-a-Judge approach.

## Features
* RAG architecture with semantic search
* ChromaDB vector database
* Ollama embeddings and LLM integration
* Source and page traceability
* AI-powered response evaluation
* Automated quality scoring
* Modular Python architecture
* Banking support simulation

## Architecture
### RAG Pipeline
1 - Banking documentation is split into chunks
2 - Embeddings are generated using nomic-embed-text
3 - Embeddings are stored in ChromaDB
4 - Customer questions are converted into embeddings
5 - Semantic similarity search retrieves relevant chunks
6 - Retrieved context is injected into the LLM prompt
7 - The assistant generates grounded responses

## Technologies

* Python
* LangChain
* ChromaDB
* Ollama
* Llama 3
* RAG (Retrieval-Augmented Generation)
* Semantic Search
* Vector Databases
* AI Evaluation Agents

## Project Structure

project/
│
├── assistant.py
├── evaluator.py
├── main.py
├── query.py
├── chroma/
├── data/
└── RAG_quality_teste.ipynb

## Assistant

The assistant is responsible for:

* Searching relevant banking information
* Answering customer questions
* Reducing hallucinations
* Using only retrieved context
* Returning grounded responses

The assistant includes:

* Semantic retrieval
* Context injection
* Prompt engineering
* Source tracking
* Page identification
* Chunk identification

## Evaluator Agent

The evaluator agent compares:

1 - Customer question
2 - Assistant response
3 - Expected response

The evaluator returns:

* 1 → Correct response
* 0 → Incorrect response

Evaluation criteria:

* Semantic similarity
* Correctness
* Relevance
* Completeness
* Intent matching

This approach simulates an automated AI quality assurance system.

## Notebook Evaluation Results

The notebook RAG_quality_teste.ipynb performs an automated evaluation pipeline containing:

* 10 banking-related questions
* 10 expected answers
* Automatic response generation
* AI-based evaluation
* Final quality scoring

## Evaluation Dataset Topics

The test set covers:

* Digital accounts
* Transfer limits
* Credit cards
* Fraud handling
* Loans
* Investments
* Insurance
* Security policies

This provides good coverage of the banking documentation and validates whether the RAG pipeline retrieves and generates accurate responses.

## Quality Results

The LLM achieved 90% accuracy during the automated evaluation process.

Results summary:

* Total questions: 10
* Correct answers: 9
* Accuracy: 90%

The evaluation was performed using a secondary LLM-based evaluator agent that compared the generated answers against expected responses using semantic similarity and intent matching.

## Example Questions

* Does the Clay Digital Account charge a monthly fee?
* What is the nighttime instant transfer limit?
* Can customers pay off the personal loan early?
* What should be done in case of fraud suspicion?

## Goals of the Project

The main goals of this project are:

* Build a realistic banking AI assistant
* Reduce hallucinations using RAG
* Improve response explainability
* Create an automated evaluation framework
* Simulate enterprise AI support systems
* Study semantic retrieval pipelines
* Evaluate LLM quality automatically
* Future Improvements

## Potential future enhancements:

* Reranking models
* Hybrid search
* Streaming responses
* Multi-agent orchestration
* Memory systems
* Human feedback evaluation
* Dashboard monitoring
* API deployment
* LangGraph integration
* Real-time analytics
* Confidence scoring
* Hallucination detection

## Author

ClayBank AI Assistant was developed as a study and experimentation project focused on:

* RAG systems
* AI evaluation pipelines
* Banking assistants
* LLM applications
* Semantic search architectures
