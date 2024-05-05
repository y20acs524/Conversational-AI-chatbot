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
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question} )
    return response

def show_history():
    for i in st.session_state.messages:
        combined_text = f"{i['role']} : {i['content']}"
        st.subheader(combined_text)
        st.divider()

def get_pdf_answer(query):
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(query)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                message = {"role": "📚 ", "content": full_response}
                st.session_state.messages.append(message)
    
def chat_with_pdf():
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        try:
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                st.success("Done")
        except:
            st.warning("Please upload PDF files and try again")
        st.button('Clear Chat History', on_click=clear_chat_history)
        sh = st.button("Show History")
    if not sh:
        st.title("Chat with PDF files 📚")
        st.write("Welcome to the chat!")
        voice_chat=st.button("Voice Chat🎙️")
        if 'messages' not in st.session_state:
            st.session_state.messages = [] 
        if voice_chat:
            input = listen()
            if input:
                st.session_state.messages.append({"role": "👩🏻‍💻", "content": input})
                with st.chat_message("user"):
                    st.write(input)
                get_pdf_answer(input)
            prompt=st.chat_input()
            
        else:
            if prompt := st.chat_input():
                st.session_state.messages.append({"role": "👩🏻‍💻", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                get_pdf_answer(prompt)
    else:
        show_history()
        
recognizer = sr.Recognizer()
def listen():
    with sr.Microphone() as source:
        with st.spinner("Listening..."):
            audio = recognizer.listen(source)
    try:
        with st.spinner("Recognizing..."):
            query = recognizer.recognize_google(audio)
            return query
    except Exception as e:
        print(e)
        return None

def chat_with_assistant():
        genai.configure(api_key=os.getenv("google_api_key"))
        st.title("Chat with Assistant 🤖 ")
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
        model=genai.GenerativeModel("gemini-pro") 
        chat = model.start_chat(history=[])
        def get_gemini_response(question):
            response=chat.send_message(question,stream=True)
            return response
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        input=st.text_area("Input: ",key="input")
        col1, col2 = st.columns([3, 1])
        with col1:
            submit=st.button("Ask the question")
        with col2:
            voice_chat=st.button("Voice Chat🎙️")
        response=""
        if (submit and input):
            st.subheader(f':red[Query : ] {input}??')
            response=get_gemini_response(input)
            st.session_state['chat_history'].append(("👩🏻‍💻 ", input))
            st.subheader(f':green[Response : ]')
            ans=""
            for chunk in response:
                ans+=chunk.text
            st.markdown(ans)
            st.session_state['chat_history'].append(("🤖 ", ans))
            st.session_state['chat_history'].append(("","-"*140))
        if voice_chat:
            input = listen()
            if input:
                response = get_gemini_response(input)
                st.subheader(f':red[Query : ] {input}??')
                st.session_state['chat_history'].append(("👩🏻‍💻 ", input))
            st.subheader(f':green[Response : ]')
            ans=""
            for chunk in response:
                ans+=chunk.text
            st.markdown(ans)
            st.session_state['chat_history'].append(("🤖 ", ans))
            st.session_state['chat_history'].append(("","-"*140))
        
        if st.sidebar.button('Show History'):
            st.subheader("The Chat History is")  
            for role, text in st.session_state['chat_history']:
                st.markdown(f"{role}: {text}")  

# Function to get weather data
def get_weather(city_name):
    api_key = "open_weather_api_key"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city_name}"
    response = requests.get(url)
    data = response.json()
    return data

def fetch_news(query):
    NEWS_API_KEY = 'news_api_key'
    NEWS_API_URL = 'https://newsapi.org/v2/everything'
    params = {
        'apiKey': NEWS_API_KEY,
        'q': query,
        'language': 'en',
        'pageSize': 10  
    }
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()
    return data['articles']

def other_features(selected):
    if selected=="weather":
        st.title("Today's Weather")
        city_name = st.text_input("Enter city name", "New York")
        if st.button("Get Weather"):
            if city_name:
                weather_data = get_weather(city_name)
                if "error" not in weather_data:
                    st.subheader(f"Weather in {city_name}: {weather_data['current']['condition']['text']}")
                    st.subheader(f"Temperature: {weather_data['current']['temp_c']}°C")
                    st.subheader(f"Humidity: {weather_data['current']['humidity']}%")
                    st.subheader(f"Wind Speed: {weather_data['current']['wind_kph']} km/h")
                    st.balloons()
                else:
                    st.error(f"Error: {weather_data['error']['message']}")
            else:
                st.warning("Please enter a city name.")
    elif selected=="Global News":
        st.title("News Chatbot")
        query = st.text_input("which topic you wanna know about??")
        if query:
            articles = fetch_news(query)
            if not articles:
                st.write("No articles found.")
            else:
                for article in articles:
                    st.write(f"**{article['title']}**")
                    st.write(f"Source: {article['source']['name']}")
                    st.write(f"Published At: {article['publishedAt']}")
                    st.write(article['description'])
                    st.write(article['url'])
                    st.write('---')
    else:
        st.header("select a option using dropdown menu..!!")
        
        
def chat_with_both():
    st.title("Chat with PDF's 📚 & Assistant 🤖")
    pdf_docs = st.sidebar.file_uploader(
                "Provide additional knowledge through file uploader", accept_multiple_files=True)
    try:
        if st.sidebar.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
    except:
            st.warning("Please upload PDF files and try again")
    sh=st.sidebar.button("show history")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    if not sh:
        if 'messages' not in st.session_state:
            st.session_state.messages = [] 
        if submit := st.button("Voice Chat 🎙️"):
            prompt=listen()
            st.session_state.messages.append({"role": "👩🏻‍💻", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
        elif prompt := st.chat_input():
            st.session_state.messages.append({"role": "👩🏻‍💻", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
        if submit or prompt:
            with st.spinner("Thinking..."):
                    st.subheader("Response Through Additional Knowledge 📚")
                    response = user_input(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            if response is not None:
                message = {"role": "📚 ", "content": full_response}
                st.session_state.messages.append(message)
            st.subheader("Response Given By Assistant 🤖")
            model=genai.GenerativeModel("gemini-pro") 
            chat = model.start_chat(history=[])
            def get_gemini_response(question):
                response=chat.send_message(question,stream=True)
                return response
            
            response=get_gemini_response(prompt)
            for chunk in response:
                st.markdown(chunk.text)
            if response is not None:
                msg=""
                for chunk in response:
                    msg+=chunk.text
                message={"role":"🤖 ","content":msg}
                st.session_state.messages.append(message)
            if not prompt:
                prompt=st.chat_input()
    else:
        show_history()
            
            
def display_widget(selected):
    if selected=="chat with pdf's":
        chat_with_pdf()
    elif selected=="chat with assistant":
        chat_with_assistant()
    elif selected=="Integration":
        chat_with_both()
    elif selected=="others":
        others_selected=st.sidebar.selectbox('select one from below',('weather','Global News'),index=None)
        other_features(others_selected)
    else:
        st.title("Welcome to streamlit chat application")

def main():
    st.set_page_config(
        page_title="project Chatbot",    
    )
    set_background_picture("https://i.imgur.com/vqNVOOG.jpg")
    st.sidebar.header("select from below")
    selected=st.sidebar.radio("",["chat with pdf's","chat with assistant","Integration","others"],index=None)
    display_widget(selected)

if __name__ == "__main__":
    main()
    
    

