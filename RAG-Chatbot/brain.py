import re
from io import BytesIO
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader


# Parse the PDF content
def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            # Clean up hyphenated words and newlines
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)
    return output, filename


# Convert text to LangChain documents with metadata
def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    # Create documents for each page
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split documents into smaller chunks
    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc_chunk = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source": f"{doc.metadata['page']}-{i}",
                    "filename": filename,
                },
            )
            doc_chunks.append(doc_chunk)
    return doc_chunks


# Convert documents to a FAISS index
def docs_to_index(docs: List[Document], huggingface_model_name: str):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_model_name)
    # Create FAISS index
    index = FAISS.from_documents(docs, embeddings)
    return index


# Generate FAISS index for multiple PDFs
def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str], huggingface_model_name: str):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))
    index = docs_to_index(documents, huggingface_model_name)
    return index


# Example usage:
# Define your HuggingFace model
huggingface_model_name = "bert-base-uncased"

# Example PDFs and names (replace with actual binary PDF data and filenames)
pdf_files = [b"binary content of pdf1", b"binary content of pdf2"]  # Replace with actual PDF binary data
pdf_names = ["example1.pdf", "example2.pdf"]

# Create the FAISS index
index = get_index_for_pdf(pdf_files, pdf_names, huggingface_model_name)

# Serialize the FAISS index (optional)
with open("faiss_index.pkl", "wb") as f:
    pickle.dump(index, f)
