
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

# Load the local embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunks = []
index = None


def process_pdf(file_path):
    global chunks, index

    if not file_path:
        return "Please upload a PDF first."

    try:
        reader = PdfReader(file_path)
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        if not text.strip():
            return "No readable text found in this PDF."

        # Split text into overlapping chunks
        chunk_size = 500
        overlap = 100
        chunks = []

        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)

        # Create embeddings and FAISS index
        embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        return f"PDF ready! {len(chunks)} text chunks indexed."

    except Exception as e:
        chunks = []
        index = None
        return f"Could not process PDF: {str(e)}"


def answer_question(question):
    if not question or not question.strip():
        return "Please enter a question."

    if index is None or not chunks:
        return "Please upload and process a PDF first."

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        min(3, len(chunks))
    )

    relevant_chunks = [
        chunks[i] for i in indices[0] if i >= 0
    ]

    # Extract sentences and rank by keyword overlap
    sentences = []
    for chunk in relevant_chunks:
        sentences.extend(
            re.split(r"(?<=[.!?])\s+", chunk)
        )

    question_words = set(
        re.findall(r"\b\w+\b", question.lower())
    )

    ranked = []
    for sentence in sentences:
        words = set(re.findall(r"\b\w+\b", sentence.lower()))
        score = len(question_words & words)

        if score > 0 and len(sentence.strip()) > 15:
            ranked.append((score, sentence.strip()))

    ranked.sort(key=lambda item: item[0], reverse=True)

    answers = []
    for score, sentence in ranked:
        if sentence not in answers:
            answers.append(sentence)
        if len(answers) == 3:
            break

    if not answers:
        return (
            "I couldn't find a clear answer in the PDF. "
            "Try asking in a different way."
        )

    return "\n\n".join(answers)


with gr.Blocks(title="AI PDF RAG Chatbot") as demo:
    gr.Markdown("# AI PDF RAG Chatbot")
    gr.Markdown(
        "Upload a PDF and ask questions about its contents."
    )

    pdf_file = gr.File(
        label="Upload your PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button("Process PDF")
    status = gr.Textbox(
        label="Document Status",
        interactive=False
    )

    process_button.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )

    gr.Markdown("## Ask your document")

    question = gr.Textbox(
        label="Your question",
        placeholder="Ask something about your PDF..."
    )

    ask_button = gr.Button("Ask Question")
    answer = gr.Textbox(
        label="Answer",
        lines=6,
        interactive=False
    )

    ask_button.click(
        fn=answer_question,
        inputs=question,
        outputs=answer
    )

    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=answer
    )


if __name__ == "__main__":
    demo.launch()
