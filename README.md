# 🔎 Experiment 7 — Semantic Search + Extractive QA

A Natural Language Processing project that combines **semantic search using sentence embeddings** with **extractive Question Answering (QA)**.

The system searches across a collection of five Data Science documents, identifies the document most semantically relevant to a user's question, and then extracts the answer from that document using a pretrained extractive QA model.

---

## 📌 Project Overview

Traditional keyword-based search can struggle when the wording of a user's question differs from the wording used in a document.

This project uses **true sentence embeddings** to understand the semantic meaning of the question and documents.

The system performs two major tasks:

1. **Semantic Search**
   - Converts documents and the user's question into vector embeddings.
   - Calculates cosine similarity between the question embedding and document embeddings.
   - Selects the most relevant document.

2. **Extractive Question Answering**
   - Passes the user's question and the most relevant document to an extractive QA model.
   - Identifies and extracts the answer directly from the selected document.

---

## 🚀 Features

- Semantic search using true sentence embeddings
- Cosine similarity-based document retrieval
- Five Data Science documents
- Extractive Question Answering
- Displays the most relevant document
- Displays cosine similarity score
- Displays extracted answer
- Displays QA score
- Displays similarity scores for all documents
- Interactive Gradio frontend
- Deployable as a web application

---

## 🧠 System Architecture

```text
                User Question
                     │
                     ▼
          Sentence Transformer
       multi-qa-MiniLM-L6-cos-v1
                     │
                     ▼
            Question Embedding
                     │
                     ▼
             Cosine Similarity
                     │
          ┌──────────┴──────────┐
          │                     │
     Document 1             Document 5
     Embedding              Embedding
          │                     │
          └──────────┬──────────┘
                     ▼
           Most Relevant Document
                     │
                     ▼
              Extractive QA
                  DistilBERT
                     │
                     ▼
             Extracted Answer
                     │
                     ▼
                Gradio UI
