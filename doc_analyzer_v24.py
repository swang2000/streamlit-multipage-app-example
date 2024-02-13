""" 
This version, I started to build the app
1. Move all supportive functions into utility.py
2. Use option_menu
3. Start to use concurrent futures for multiprocessing
4. Use serialization to pass the data between processes
5. Move summarization processes into side bar
6. The version 2.3 has a kind of multiple processes, but it runs in Q&A again and again
7. Comparing with 2.3, the version of 2.4 is trying to resolve the multiple process issue. 
8. Summarization block is moved to summarize section. It is not multiprocessing yet. It is just static 
 


"""

import os
# os.chdir('./Doc_analyer')
#print(os.getcwd())
import tempfile
import streamlit as st
from streamlit_option_menu import option_menu
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from datetime import datetime
#from utils.utility_sai import summarizer
#from summarization_with_MR import summarizer
from utils.utility_v24 import PrintRetrievalHandler, StreamHandler, configure_retriever, summarizer, summary_section


EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')


#ChatGPT credentials
import openai
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai_deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
openai_embedding_model_name = os.getenv('OPENAI_EMBEDDING_MODEL_NAME')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version =  os.getenv('OPENAI_API_VERSION')
MODEL_NAME = os.getenv('MODEL_NAME')

config = dotenv_values(".env")





if __name__ == '__main__':

    st.set_page_config(page_title="DMI - iMedical insights (Genie)", page_icon="🦜", layout='wide')
    
    
    
    
    if 'first_two_words' not in st.session_state:
        st.session_state['first_two_words'] = []
    if 'doc_name' not in st.session_state:
        st.session_state['doc_name'] = []
    if 'doc_summary' not in st.session_state:
        st.session_state['doc_summary'] = []
    if 'files_uploaded' not in st.session_state:
        st.session_state['files_uploaded'] = []
    if 'summarized_doc' not in st.session_state:
        st.session_state['summarized_doc'] = []
    if 'summarize_start' not in st.session_state:
        st.session_state['summarize_start'] = False
    if 'file_names' not in st.session_state:
        st.session_state['file_names'] = []
    if 'doc_list' not in st.session_state:
        st.session_state['doc_list'] = []

    st.session_state['files_uploaded'] = st.sidebar.file_uploader(
        label="Upload pdf, word, or txt files", type=["pdf", "doc", "docx", "txt",  "html", "ppt", "pptx"], accept_multiple_files=True
    )
    if len(st.session_state['files_uploaded'])==0:
        st.info("Please upload pdf, word, or txt, ppt, or pptx documents to continue.")
        st.stop()
    retriever, docs, documents_list = configure_retriever(st.session_state['files_uploaded'])


    st.session_state['file_names'] = [file.name for file in st.session_state['files_uploaded']]
    st.session_state['doc_name'] = [file['name'] for file in documents_list]
    st.session_state['doc_list'] = documents_list
    
    
    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        st.write("")
    with col2:
        st.image("The_Genie_Aladdin_DMI.png")
        #st.write(config)
    with col3:
        st.write("")

    
    

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        # model_name="gpt-3.5-turbo-16k", engine = MODEL_NAME, openai_api_key=openai.api_key, temperature=0, streaming=True
        model=config["MODEL_NAME"],
        engine=config["MODEL_NAME"],
        openai_api_key=config["OPENAI_API_KEY"],
        temperature=0,
        openai_api_base=config["OPENAI_API_BASE"],  streaming=True,
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    #st.write(openai.api_version)

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)
    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        
        
        

                      


        
    
    
    

    
    #print('Here is the session state data: ', st.session_state['level1'])
    




    

    
