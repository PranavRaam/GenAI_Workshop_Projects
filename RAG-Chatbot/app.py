import databutton as db
import streamlit as st
from brain import get_index_for_pdf
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import requests  # Added missing import
from langchain.embeddings import HuggingFaceEmbeddings

# Hugging Face API Token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Hugging Face API token is missing. Set it as an environment variable.")

HUGGINGFACE_MODEL_NAME = "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"

# Streamlit Title
st.title("RAG Enhanced Chatbot with Hugging Face Models")

# Define function to call Hugging Face Inference API
def query_huggingface_model(input_text):
    """
    Call Hugging Face Inference API to generate a response from the model.
    """
    url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL_NAME}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    }
    data = {
        "inputs": input_text,
        "parameters": {
            "max_length": 500,
            "temperature": 0.7,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise HTTP errors if any
        return response.json()[0].get("generated_text", "Error: No generated text.")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Cache the creation of the vector database
@st.cache_data
def create_vectordb(files, filenames):
    """
    Create a vector database for the uploaded PDF files.
    """
    with st.spinner("Creating vector database..."):
        embeddings = HuggingFaceEmbeddings()
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, embeddings
        )
    return vectordb

# Function to process user questions
def process_question(question, vectordb, prompt_template):
    """
    Process the user's question and generate a response.
    """
    # Search the vector database for relevant context
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Prepare the prompt for the model
    system_prompt = prompt_template.format(pdf_extract=pdf_extract)
    input_prompt = f"{system_prompt}\nUser: {question}\nAssistant:"

    # Use Hugging Face Inference API to generate the response
    response = query_huggingface_model(input_prompt)
    return response

# File uploader for PDFs
pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# If PDFs are uploaded, create a vector database
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    if "vectordb" not in st.session_state:
        st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Prompt template for the assistant
prompt_template = """
    You are a helpful assistant who answers users' questions based on multiple contexts given to you.

    Keep your answer short and to the point.

    The evidence is the context of the PDF extract with metadata. 

    Carefully focus on the metadata, especially 'filename' and 'page' whenever answering.

    Make sure to add filename and page number at the end of the sentence you are citing.

    Reply "Not applicable" if the text is irrelevant.

    The PDF content is:
    {pdf_extract}
"""

# Chat interaction
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

# Display previous chat messages
for message in st.session_state["prompt"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input for user question
question = st.chat_input("Ask anything")

# Handle user input
if question:
    vectordb = st.session_state.get("vectordb", None)

    if not vectordb:
        # Notify the user if no PDFs are uploaded
        with st.chat_message("assistant"):
            st.write("Please upload at least one PDF to proceed.")
    else:
        # Process the question and generate a response
        response = process_question(question, vectordb, prompt_template)

        # Display the conversation
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(response)

        # Update the session state for chat history
        st.session_state["prompt"].append({"role": "user", "content": question})
        st.session_state["prompt"].append({"role": "assistant", "content": response})
