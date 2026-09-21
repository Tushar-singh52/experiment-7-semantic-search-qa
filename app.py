import os
from pathlib import Path

import gradio as gr
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================

BASE_DIR = Path(__file__).parent

documents = {}

for i in range(1, 6):

    filename = f"document{i}.txt"
    file_path = BASE_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        documents[filename] = file.read()


document_names = list(documents.keys())
document_texts = list(documents.values())


# ============================================================
# 2. MODEL VARIABLES
# ============================================================

embedding_model = None
document_embeddings = None

qa_tokenizer = None
qa_model = None


# ============================================================
# 3. LOAD MODELS ONLY WHEN NEEDED
# ============================================================

def load_models():

    global embedding_model
    global document_embeddings
    global qa_tokenizer
    global qa_model

    # Load semantic search model
    if embedding_model is None:

        print("Loading semantic search model...")

        embedding_model = SentenceTransformer(
            "multi-qa-MiniLM-L6-cos-v1"
        )

        document_embeddings = embedding_model.encode(
            document_texts
        )

        print("Semantic search model loaded.")


    # Load extractive QA model
    if qa_model is None:

        print("Loading extractive QA model...")

        qa_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-cased-distilled-squad"
        )

        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-cased-distilled-squad"
        )

        print("Extractive QA model loaded.")


# ============================================================
# 4. SEMANTIC SEARCH
# ============================================================

def semantic_search(question):

    question_embedding = embedding_model.encode(
        [question]
    )

    similarity_scores = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    best_index = similarity_scores.argmax()

    best_document_name = document_names[best_index]
    best_document = document_texts[best_index]
    best_score = similarity_scores[best_index]

    return (
        best_document_name,
        best_document,
        best_score,
        similarity_scores
    )


# ============================================================
# 5. EXTRACTIVE QUESTION ANSWERING
# ============================================================

def extract_answer(question, context):

    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop(
        "offset_mapping"
    )[0]

    sequence_ids = inputs.sequence_ids(0)

    with torch.no_grad():

        outputs = qa_model(
            **inputs
        )

    start_probs = torch.softmax(
        outputs.start_logits,
        dim=-1
    )[0]

    end_probs = torch.softmax(
        outputs.end_logits,
        dim=-1
    )[0]

    context_indices = [
        i
        for i, sid in enumerate(sequence_ids)
        if sid == 1
    ]

    best_score = 0

    best_start = context_indices[0]
    best_end = context_indices[0]

    for start in context_indices:

        for end in context_indices:

            if end < start:
                continue

            if end - start > 20:
                continue

            score = (
                start_probs[start].item()
                *
                end_probs[end].item()
            )

            if score > best_score:

                best_score = score
                best_start = start
                best_end = end

    start_char = offset_mapping[
        best_start
    ][0].item()

    end_char = offset_mapping[
        best_end
    ][1].item()

    answer = context[
        start_char:end_char
    ]

    return answer, best_score


# ============================================================
# 6. COMPLETE QUESTION-ANSWERING PIPELINE
# ============================================================

def answer_question(question):

    if not question.strip():

        return (
            "",
            "",
            "",
            "",
            ""
        )

    # Load models when first question is submitted
    load_models()

    (
        best_document_name,
        best_document,
        best_similarity,
        scores
    ) = semantic_search(question)

    answer, qa_score = extract_answer(
        question,
        best_document
    )

    similarity_output = ""

    for name, score in zip(
        document_names,
        scores
    ):

        similarity_output += (
            f"{name}: {score:.4f}\n"
        )

    return (
        best_document_name,
        f"{best_similarity:.4f}",
        answer,
        f"{qa_score:.4f}",
        similarity_output
    )


# ============================================================
# 7. GRADIO FRONTEND
# ============================================================

with gr.Blocks(
    title="Experiment 7 - Semantic Search + Extractive QA"
) as demo:

    gr.Markdown(
        """
        # 🔎 Semantic Search + Extractive QA

        ### Experiment 7

        Enter a question to find the most relevant
        document using semantic search and extract
        the answer using an extractive Question
        Answering model.
        """
    )

    question = gr.Textbox(
        label="Enter your question",
        placeholder="Example: What is data preprocessing?",
        lines=2
    )

    search_button = gr.Button(
        "Search",
        variant="primary"
    )

    with gr.Row():

        document_output = gr.Textbox(
            label="Most Relevant Document"
        )

        similarity_output = gr.Textbox(
            label="Cosine Similarity"
        )

    answer_output = gr.Textbox(
        label="Extracted Answer",
        lines=3
    )

    qa_score_output = gr.Textbox(
        label="QA Score"
    )

    scores_output = gr.Textbox(
        label="Similarity Scores",
        lines=6
    )

    search_button.click(
        fn=answer_question,
        inputs=question,
        outputs=[
            document_output,
            similarity_output,
            answer_output,
            qa_score_output,
            scores_output
        ]
    )


# ============================================================
# 8. START GRADIO SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    print(f"Starting Gradio server on port {port}...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )