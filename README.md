# AI-TRAVEL-PLANNER
import os
from dotenv import load_dotenv
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.6
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["destination", "days", "budget", "travel_type"],
    template="""
    You are a smart travel planner.

    Create a detailed day-wise travel itinerary.

    Destination: {destination}
    Number of Days: {days}
    Budget: {budget}
    Travel Type: {travel_type}

    Include:
    - Places to visit each day
    - Suggested activities
    - Food recommendations
    - Budget-friendly tips
    """
)

# LangChain
chain = LLMChain(llm=llm, prompt=prompt)

# Function for Gradio
def generate_plan(destination, days, budget, travel_type):
    response = chain.run(
        destination=destination,
        days=days,
        budget=budget,
        travel_type=travel_type
    )
    return response

# Gradio UI
interface = gr.Interface(
    fn=generate_plan,
    inputs=[
        gr.Textbox(label="Destination"),
        gr.Textbox(label="Number of Days"),
        gr.Textbox(label="Budget"),
        gr.Dropdown(
            ["Solo", "Family", "Friends", "Couple"],
            label="Travel Type"
        )
    ],
    outputs=gr.Textbox(label="Your Travel Plan"),
    title="AI Travel Planner",
    description="Generate a personalized travel itinerary using AI"
)

# Run app
interface.launch()

