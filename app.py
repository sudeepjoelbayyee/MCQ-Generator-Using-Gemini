import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from utils import read_file, get_table_data
import streamlit as st
from mcq_generator import generate_evaluate_chain

with open('response.json','r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ Generator With Gemini LangChain")

with st.form("User Input"):
    fileupload = st.file_uploader("Upload PDF/TXT")

    number = st.number_input("Number of MCQ's",min_value=3,max_value=50)

    subject = st.text_input("What is the Subject of PDF content you Uploaded", max_chars=20)

    tone = st.text_input("Complexity Level of questions",max_chars=20,placeholder='simple')

    button = st.form_submit_button("Create MCQ's")

if button and fileupload is not None and number and subject and tone:
    with st.spinner("loading..."):
        try:
            text = read_file(fileupload)
            response = generate_evaluate_chain.invoke({
            "text": text,
            "number": number,
            "subject": subject,
            "tone": tone,
            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
                }
        )
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")
        else:
            if isinstance(response,dict):
                quiz = response.get("quiz",None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)

                        st.text_area(label="Review",value=response['review'],height=350)
                    else:
                        st.error("Error in the table data")
                else:
                    st.write(response)

        
