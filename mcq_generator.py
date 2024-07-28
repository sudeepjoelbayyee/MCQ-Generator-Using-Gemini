from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import json
import os
import pandas as pd
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),model="gemini-pro",temperature=0.7)

with open("response.json",'r') as f:
    RESPONSE_JSON = json.load(f)

TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. Make sure the questions are not repeated and check all the questions to be conforming the text as well. Make sure to format your response like {RESPONSE_JSON} below and use it as a guide. Ensure to make {number} MCQs.
"""

TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students. You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. If the quiz is not at per with the cognitive and analytical abilities of the students, update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}
Check from an expert English Writer of the above quiz:
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
    )

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], 
        input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
        output_variables=["quiz", "review"], verbose=True)