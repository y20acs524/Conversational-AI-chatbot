import os
import speech_recognition as sr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests 

load_dotenv()
os.getenv("google_api_key")
genai.configure(api_key=os.getenv("google_api_key"))


def set_background_picture(url):
    st.markdown(
         f"""
         <style>
         [data-testid="stAppViewContainer"] {{
             background: url("{url}");
             background-size: cover;
         }}
         [data-testid=stSidebar]{{
              background: url("{url}");
              background-size: cover;
         }}
         [data-testid=stHeader]{{
              background: url("{url}");
             background-size: cover;

         }}
         [data-testid=stBottomBlockContainer]{{
             background: url("https://i.imgur.com/EN8aa2k.jpg");
             background-size: cover;
             padding: 10px;
             border-radius: 500px;
         }}
         </style>
         """,
         unsafe_allow_html=True
        )

# read all pdf files and return text
def get_pdf_text(pdf_docs):
.....//mail me for code requests or just msg me on whatsapp(8179313727)//
    
    

