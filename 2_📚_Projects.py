import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed



def workerM(t):
        
        counter = 0
        while counter > t:
            time.sleep(1)
            counter -= 1
            print(counter)
        


    
    

    
st.title("Projects")

st.write("You have entered", st.session_state["my_input"]) 

if st.session_state['project']:
    st.session_state['project'] = False
    with ThreadPoolExecutor() as executor:
        pj = executor.submit(workerM, -20)
    print(pj.result())
