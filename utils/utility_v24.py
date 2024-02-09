"""
This version is to add the following fuctions:
1. Import data from different sources - pdf, doc, ppt
    Add pptx package and import from ppt
    Move from pdftotext, doc2txt to langchain. Major challenge here is that environment to import unstructured ppt and pdf - poppler, config packages 
    Installation of additional packages

2. Explore the summarization for long doc 
    Explore the different options - map-reduce, re-rank, refine 
    Explore the chunking size

"""


import os, openai
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, dotenv_values
from langchain.document_loaders import Docx2txtLoader
import os
from PIL import Image
import re
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from concurrent.futures import ProcessPoolExecutor, as_completed



from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    GitLoader,
    NotebookLoader,
    OnlinePDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEPubLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredODTLoader,
    NotebookLoader,
    UnstructuredFileLoader
)

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


FILE_LOADER_MAPPING = {
   "csv": (CSVLoader, {"encoding": "utf-8"}),
    "doc": (Docx2txtLoader, {}),
    "docx": (Docx2txtLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "pdf": (PyPDFLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "csv": (UnstructuredCSVLoader, {}),
    "xlsx": (UnstructuredExcelLoader, {"mode": "elements", "autodetect_encoding": "True"}),
    "xls": (UnstructuredExcelLoader, {"mode": "elements"}),
    "txt": (TextLoader, {"encoding": "utf-8"}),
 
}

load_dotenv()
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE = os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_API_VERSION')
EMDEDDING_ENGINE = os.getenv('EMDEDDING_ENGINE')
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
openai.api_type = os.getenv('OPENAI_API_TYPE')
MODEL_NAME = os.getenv('MODEL_NAME')


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
# Read documents
    docs = []
    docs_list= []
    file_names = []
    temp_dir = tempfile.TemporaryDirectory()
    for uploaded_file in uploaded_files:
               
                ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
                file_names.append(uploaded_file.name)

                # Check if the extension is in FILE_LOADER_MAPPING
                if ext in FILE_LOADER_MAPPING:
                    loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                    # st.write(f"loader_class: {loader_class}")

                    # Save the uploaded file to the temporary directory
                    file_path = os.path.join(temp_dir.name, uploaded_file.name)
                    with open(file_path, 'wb') as temp_file:
                        temp_file.write(uploaded_file.read())

                    # Use Langchain loader to process the file
                    loader = loader_class(file_path, **loader_args)
                    docs.extend(loader.load())
                    # docs_list.append(loader.load())
                    docs_list.append({'name': uploaded_file.name, 'content': loader.load()})
                else:
                    st.warning(f"Unsupported file extension: {ext}")


    # for file in uploaded_files:
    #     temp_filepath = os.path.join(temp_dir.name, file.name)
    #     with open(temp_filepath, "wb") as f:
    #         f.write(file.getvalue())
    #     loader = PyPDFLoader(temp_filepath)
    #     docs.extend(loader.load())

    # Split documents
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    '''# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name= 'cl100k_base', separators = ["\n\n"], keep_separator = False, chunk_size=2500, chunk_overlap=100)'''
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, chunk_overlap=0 )
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(model = EMDEDDING_ENGINE, openai_api_key=EMBEDDING_API_KEY,  \
            openai_api_base=EMBEDDING_API_BASE,
            openai_api_type= openai.api_type,
            chunk_size=5) 
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 4})

    return retriever, docs, docs_list


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def splitfile(docs):
    text_dict = {}
    for i in range(len(docs)):
        text = docs[i].page_content
        text = text.replace("\n", "").replace("'", "").replace("?", "? ")
        text_list = re.split(r"\[//.+?//\]", text)
        #sections = re.findall(r"\[//.+?//\]", text)
        sections = [
            x.replace("[//", "").replace("//]", "")
            for x in re.findall(r"\[//.+?//\]", text)
        ]
        text_list = [x for x in text_list if len(x) > 200]
        print(sections)
        print(len(text_list))
        if len(sections) != len(text_list):
            print('Something wrong is with docx reading in')
        else:
            for j in range(len(text_list)):
                if sections[j] not in text_dict.keys():
                    text_dict[sections[j]] = text_list[j]
                else:
                    text_dict[sections[j]] = text_dict[sections[j]] +'\n\n '+text_list[j]
    return text_dict

@st.cache_resource
def prepareText(files):
    print(files)
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    print('uploaded_files are {files}')
    for file in files:
        print(f'The file was printed: {file}')
        
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = Docx2txtLoader(temp_filepath)
        docs.extend(loader.load())
    print(docs)
    text_dict = splitfile(docs)
    return text_dict

def summarizer(docs):
    
    doc_content = ''.join([p.page_content for p in docs]) 
    # st.write(doc_content)
    #print('The passed doc lengths: ', len(docs))
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=10000, chunk_overlap=0,  separators = ["\n\n", "\n"]  )
    
    doc = text_splitter.split_text(doc_content)
    # st.write(texts)

    map_template_string = """The following is a set of docs:  
    {text}
    Based on this set of docs, You are tasked with summarizing the docs at hand into a list of distinct topics with a high level of detail. For each topic you identify, provide a comprehensive summary that captures the essence of the discussion around it. 
    """
    reduce_template_string = """The following is set of summaries:
    {text}
    Your goal is to summarize these summaries into a list of topics. For each topic, provide a well-crafted summary.  Please don't miss any topic in the set of summaries provided"""




    map_prompt = PromptTemplate(input_variables=["text"], template=map_template_string)
    reduce_prompt = PromptTemplate(input_variables=["text"], template=reduce_template_string)

    #llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", engine = MODEL_NAME)
    llm = ChatOpenAI(
            # model_name="gpt-3.5-turbo-16k", engine = MODEL_NAME, openai_api_key=openai.api_key, temperature=0, streaming=True
            model=config["MODEL_NAME"],
            engine=config["MODEL_NAME"],
            openai_api_key=config["OPENAI_API_KEY"],
            temperature=0,
            openai_api_base=config["OPENAI_API_BASE"],  streaming=True,
        )
    
    chain = load_summarize_chain(llm, chain_type="map_reduce",  map_prompt = map_prompt, combine_prompt = reduce_prompt)
    #texts   map_reduce
    documents = [Document(page_content=t) for t in doc]
    print('The documents are summarized with: ', len(documents))
    #print(documents[-1])
    concise_summary = chain.run(documents)

    return concise_summary

# Function to handle the summarization of documents
def summary_section(docs_list):
        # Check if the number of uploaded files exceeds 5

    
        # if docs_list and len(docs_list) > 5: #selected options must not be greater than 5
        #     st.warning("Please upload only up to 5 documents.")
            
        # else:
        if len(docs_list)<=5:   
            doc_summary = []
            first_two_words = []
            doc_name = []
            for index, document in enumerate(docs_list):
                    # Parallel processing for summarizing documents
                    with ProcessPoolExecutor(max_workers=len(docs_list)) as executor:
                            pj = executor.submit(summarizer, document["content"])
                            summary = pj.result()
                            doc_summary.append(summary)
                            doc_name.append(document["name"])
                            first_two_words.append(' '.join(document['name'].split()[:2])+'_'+str(index+1))

            return first_two_words, doc_name, doc_summary

_ ="""
'search' function which is intended to find answers to specific questions using a given context
This function uses an instance of ChatOpenAI configured with the GPT-3.5 Turbo model. The core mechanism involves constructing and executing a map-reduce chain specifically designed for question-answering tasks.
When a non-empty question is provided, the function generates a search query using two distinct prompt templates: one for mapping and another for combining.
The map-reduce chain along with load_qa_with_sources_chain, processes the input context and the question to generate an answer.
"""

        



