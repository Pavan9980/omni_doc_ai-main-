import streamlit as st
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

# Download punkt tokenizer if not already present
nltk.download('punkt')

LANGUAGE = "english"
SENTENCES_COUNT = 5  # Number of sentences in summary

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, SENTENCES_COUNT)
    return " ".join(str(sentence) for sentence in summary)

def main():
    st.title("PDF Summarizer using Sumy (Extractive)")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_pdf)

        if pdf_text:
            st.write("### Extracted Text Preview:")
            st.write(pdf_text[:1000] + " ...")

            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(pdf_text)
                st.write("### Summary:")
                st.write(summary)
        else:
            st.error("Failed to extract text from PDF.")

if __name__ == "__main__":
    main()

#streamlit run c:/Users/WELCOME/Desktop/pdf_chatbot/app1.py
