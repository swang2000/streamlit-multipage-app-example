import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

def worker(t):
        
        counter = 0
        while counter < t:
            time.sleep(1)
            counter += 1
            print(counter)
        


    
    

    

if __name__ == '__main__':
    st.title("Main Page")
    st.sidebar.success("Select a page above.")
    if 'count' not in st.session_state:
         st.session_state['count'] = True

    if 'project' not in st.session_state:
         st.session_state['project'] = True

    if "my_input" not in st.session_state:
        st.session_state["my_input"] = ""

    my_input = st.text_input("Input a text here", st.session_state["my_input"])
    submit = st.button("Submit")
    if submit:
        st.session_state["my_input"] = my_input
        st.write("You have entered: ", my_input)
    
    if st.session_state['count']==True:
        st.session_state['count'] = False
        with ThreadPoolExecutor() as executor:
            pj = executor.submit(worker, 30)
        print(pj.result())
    
    


    