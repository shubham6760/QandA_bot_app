import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

# Load T5 model and tokenizer
model_name = "t5-small"  # You can change this to another T5 model if desired
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate response using T5 model
def generate_response(query, document):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user query in chat message container
    with st.container():
        st.write("User Query: ", query)

    # If document is provided, process it
    if document is not None:
        file_type = document.name.split(".")[-1]
        if file_type == "pdf":
            text = read_pdf(document)
        elif file_type == "csv":
            text = read_csv(document)
        elif file_type == "docx":
            text = read_docx(document)
        else:
            text = ""
    else:
        text = ""

    # Generate response using T5 model
    input_text = "chatbot: " + query + " " + text + " </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response from model
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display response in chat message container
    with st.container():
        st.write("Assistant: ", response)

# Function to read text from PDF
def read_pdf(document):
    pdf_reader = PdfReader(document)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read text from CSV
def read_csv(document):
    df = pd.read_csv(document, encoding="latin-1")  # Specify 'latin-1' encoding
    text = "\n".join(df.iloc[:, 0].astype(str))
    return text

# Function to read text from DOCX
def read_docx(document):
    doc = Document(document)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main application code
st.title("Shubham Raj Assignment Chatbot")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.write("User: ", message["content"])
        elif message["role"] == "assistant":
            st.write("Assistant: ", message["content"])

# Accept user input (query and document)
query = st.text_input("Enter your query:")
document = st.file_uploader("Upload a document (PDF, CSV, or DOCX):", type=["pdf", "csv", "docx"])

if st.button("Ask"):
    if query:
        generate_response(query, document)
    else:
        st.warning("Please enter a query.")

# Save chat history
if st.button("Save Chat History"):
    with open("chat_history.txt", "w") as file:
        for message in st.session_state.messages:
            file.write(f"{message['role']}: {message['content']}\n")
    st.success("Chat history saved successfully.")
