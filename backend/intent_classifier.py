from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file")

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Request and Response models
class ClassifyRequest(BaseModel):
    query: str
    selected_role: str  # from dropdown

class ClassifyResponse(BaseModel):
    intent: str
    valid: bool
    message: str

# LangChain LLM setup
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Prompt template
prompt_template = PromptTemplate(
    template="""You are an expert intent classifier.

Classify the following text into one of the following categories:
[coding, software_engineering, music, recruitment, sales, general]

Examples:
1. "How do I reverse a string in Python?" → coding
2. "Explain recursion with an example." → coding
3. "How to design a scalable microservice architecture?" → software_engineering
4. "What is the difference between monolith and microservices?" → software_engineering
5. "Top 10 trending music albums of 2023." → music
6. "Play a relaxing piano playlist." → music
7. "I’m looking for a job change in data science." → recruitment
8. "Can you help with resume screening for backend roles?" → recruitment
9. "How to increase leads through email marketing?" → sales
10. "My monthly targets are not being met, suggest strategies." → sales
11. "What's the weather like in Bangalore?" → general
12. "Tell me a joke about cats." → general

If the category does not clearly fit, classify as 'general'.

Text: "{text}"
Intent:""",
    input_variables=["text"],
)

code_chain = LLMChain(llm=llm, prompt=prompt_template)

# Role to allowed intents mapping
role_to_intents = {
    "Software Engineer": ["software_engineering", "coding"],
    "Music Teacher": ["music"],
    "Salesman": ["sales"],
    "Recruiter": ["recruitment"],
    "General": ["general", "coding", "software_engineering", "music", "recruitment", "sales"]
}

@app.post("/classify", response_model=ClassifyResponse)
async def classify_intent(payload: ClassifyRequest):
    query = payload.query.strip()
    selected_role = payload.selected_role.strip()

    try:
        result = code_chain.invoke({"text": query})
        intent = result["text"].strip().lower()

        allowed_intents = role_to_intents.get(selected_role, [])
        if intent in allowed_intents:
            return {
                "intent": intent,
                "valid": True,
                "message": "Intent matches the selected role. Proceed."
            }
        else:
            return {
                "intent": intent,
                "valid": False,
                "message": "Intent does not match the selected role. Please select 'General' as the role."
            }

    except Exception as e:
        logging.error(f"Error in LLM classification: {e}")
        return {
            "intent": "unknown",
            "valid": False,
            "message": "Something went wrong while classifying the intent."
        } 