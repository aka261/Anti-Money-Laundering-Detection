import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )

def explain_transaction(transaction: dict, risk_score: float) -> str:
    prompt = PromptTemplate(
        input_variables=["transaction", "risk_score"],
        template="""
You are an AML (Anti-Money Laundering) analyst.

Analyze this transaction and explain why it is or is not suspicious.

Transaction details:
{transaction}

Risk score: {risk_score} (0 = safe, 1 = highly suspicious)

Give a short 3 to 4 sentence explanation in simple English.
Mention specific suspicious patterns if risk score is above 0.5.
"""
    )
    llm   = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transaction": str(transaction), "risk_score": risk_score})

def explain_model_results(results: dict) -> str:
    prompt = PromptTemplate(
        input_variables=["results"],
        template="""
You are a data scientist explaining model results to a business audience.

Here are the AML model comparison results:
{results}

Explain in simple English:
1. Which model performed best and why
2. Why recall is more important than accuracy for fraud detection
3. One recommendation for production deployment

Keep it short and simple, 4 to 5 sentences.
"""
    )
    llm   = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"results": str(results)})