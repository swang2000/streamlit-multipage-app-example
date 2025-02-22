import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.utility_v24 import summarizer, summary_section


if 'selected_upload' not in st.session_state:
    st.session_state['selected_upload'] = []

# def keep(key):
#     # Copy from temporary widget key to permanent key
#     st.session_state[key] = st.session_state['_'+key]

# def unkeep(key):
#     # Copy from permanent key to temporary widget key
#     st.session_state['_'+key] = st.session_state[key]
# unkeep('file_names')
# selected_upload_files = st.multiselect('Select files to Summarize(Up to 3)', st.session_state['file_names'], key='_file_names', on_change=keep, args=['file_names'])

# st.session_state['selected_upload'] = [file for file in st.session_state['doc_list'] if file["name"] in selected_upload_files]

# if selected_upload_files and len(selected_upload_files) > 3: #selected options must not be greater than 3
#     st.warning("Please upload only up to 3 documents.")

def click_button():
        st.session_state['summarize_start'] = True
        
st.write('The first check with file_name: ', st.session_state['file_names'])
st.button('start summarize...', on_click=click_button)
 
st.write('The second check with file_name: ', st.session_state['file_names'])

if st.session_state['summarize_start']:
    #st.session_state['summarize_start'] = False
    st.write(st.session_state['doc_summary'])
    st.write(set(st.session_state['summarized_doc']))
    st.write(set(st.session_state['file_names']))

    if len(st.session_state['doc_summary'])==0:  #or (set(st.session_state['summarized_doc']) != set(st.session_state['selected_upload']))
        # st.sidebar.write(st.session_state['summarized_doc'], st.session_state['doc_name'], st.session_state['doc_summary'])
        #with st.sidebar.spinner('Summarization process is running...'):
        with ThreadPoolExecutor(max_workers=1) as executor:
            st.write('Run summarization now....')
            pj = executor.submit(summary_section, st.session_state['doc_list'])
            st.session_state['first_two_words'], st.session_state['summarized_doc'], st.session_state['doc_summary'] = pj.result()
        st.write('The summarization is completed')
        
    st.sidebar.write('The summarization process is completed')

st.empty()
#st.session_state['summarize_start'] 
#st.write(len(st.session_state['doc_summary']))

if len(st.session_state['doc_summary'])==0:
    st.spinner('Running the summarization process...')
elif len(st.session_state['doc_summary'])>0:
    tabs= st.tabs(st.session_state['first_two_words'])
    for index in range(len(tabs)):
        with tabs[index]:
            st.write(st.session_state['summarized_doc'][index])
            st.write(st.session_state['doc_summary'][index])
    print('The summarization process is completed')

