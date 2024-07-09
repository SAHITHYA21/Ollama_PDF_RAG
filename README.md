# Ollama Local LLM RAG

## Introduction

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** application in Python, enabling users to query and chat with their PDFs using generative AI. The setup includes advanced topics such as running RAG apps locally with Ollama, updating a vector database with new items, using RAG with various file types, and testing the quality of AI-generated responses.

## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone https://github.com/SAHITHYA21/Ollama_PDF_RAG.git
    cd easy-local-rag
    ```

2. **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Install Ollama:**
    - Download and install Ollama from [this link](https://ollama.com/download).

4. **Pull the necessary models:**
    ```sh
    ollama pull llama3
    ollama pull mxbai-embed-large
    ollama pull dolphin-llama3
    ```

5. **Run the upload script with your files (PDF, .txt, JSON):**
    ```sh
    python upload.py
    ```

6. **Run the local RAG application:**
    - With query re-write:
        ```sh
        python localrag.py
        ```

## What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances the capabilities of **Language Learning Models (LLMs)** by combining their powerful language understanding with the targeted retrieval of relevant information from external sources. This often involves using embeddings stored in vector databases, resulting in more accurate, trustworthy, and versatile AI-powered applications.

## What is Ollama?

**Ollama** is an open-source platform that simplifies the process of running powerful LLMs locally on your own machine. It provides users with more control and flexibility in their AI projects. Learn more at [Ollama's website](https://www.ollama.com).

---

Feel free to contribute to the project or reach out with any questions or suggestions!

