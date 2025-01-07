import streamlit as st
import os
from loguru import logger
import sys
from streamlit_pdf_viewer import pdf_viewer
import time

# Set up the logger to display the log messages in the console
try:
    logger.remove(0) # Remove the default handler configuration
except ValueError:
    pass
logger.add(sys.stdout, level="DEBUG", serialize=True) # Add a new handler to display the log messages in the console


# Load the OLLAMA model and use it to answer a question about a specific section of a PDF document
from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.ollama import Ollama

# Create an instance of the OLLAMA model
llm = Ollama(model="llama3", request_timeout=240.0)
logger.debug("Model loaded successfully")

uploaded_file = st.file_uploader("Choose a file", type=(["pdf"]))
file_path = None
if uploaded_file is not None:
    file_path = os.path.join("pdf_dir/", uploaded_file.name)   
    logger.debug("PDF file uploaded")
    with open(file_path,"wb") as f: 
        f.write(uploaded_file.getbuffer())     
    st.success("Saved File: " + file_path)
    binary_data = uploaded_file.getvalue()
    pdf_viewer(input=binary_data, width=700)

    time1 = time.time()
    # Define the URL of the PDF document and the LLM-Sherpa API URL
    llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
    pdf_url = "./" + file_path

    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    logger.debug("PDF reader created successfully")

    # Read the PDF document
    doc = pdf_reader.read_pdf(pdf_url)
    logger.debug("PDF document read successfully")

    # Find the section containing the Q1 2024 Financial Highlights
    selected_section = None
    for section in doc.sections():
        if 'Q1 2024 Financial Highlights' in section.title:
            selected_section = section
            break
    logger.debug("Selected section: {}", selected_section.title)

    # Convert the output in HTML format
    context = selected_section.to_html(include_children=True, recurse=True)
    logger.debug("HTML context: {}", context)

    # Ask a question
    with st.form("ask-form"):
        question = st.text_input("Ask a question: ", "Summarize the Q1 2024 Financial Highlights")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            logger.info("Question asked: {}", question)
            with st.spinner('Wait for it...'):
                resp = llm.complete(
                    f"read this table and answer question: {question}:\n{context}")
                logger.info("Question answered successfully")

            # Show the response
            st.write("Response: " + resp.text)
            logger.debug("Response: {}", resp.text) 
            time2 = time.time()
            logger.debug("Time to analyze PDF: {}", time2 - time1)
else:
    st.write("No PDF file has been uploaded")