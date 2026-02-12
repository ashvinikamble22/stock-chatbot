import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(page_title="Financial AI Chatbot", page_icon="📊")
st.title("📊 Financial Conversational AI (FinBERT Powered)")

# -----------------------------------
# API KEY
# -----------------------------------

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# -----------------------------------
# LOAD FINBERT (Cached)
# -----------------------------------

@st.cache_resource
def load_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
labels = ["negative", "neutral", "positive"]

# -----------------------------------
# FUNCTIONS
# -----------------------------------

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    sentiment_id = torch.argmax(probs).item()
    confidence = probs[0][sentiment_id].item()
    return labels[sentiment_id], round(confidence, 3)


def get_stock_price(symbol):
    ticker = yf.Ticker(symbol.upper())
    data = ticker.history(period="1d")
    if data.empty:
        return None
    return round(data["Close"].iloc[-1], 2)


def get_financial_news(company):
    articles = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        page_size=5
    )
    return articles["articles"]


def generate_response(user_input):
    user_input_lower = user_input.lower()

    # STOCK PRICE
    if "price" in user_input_lower:
        symbol = user_input.split()[-1]
        price = get_stock_price(symbol)
        if price:
            return f"💰 Current price of {symbol.upper()} is ${price}"
        else:
            return "❌ Stock symbol not found."

    # NEWS + SENTIMENT
    elif "news" in user_input_lower:
        company = user_input.replace("news", "").strip()

        articles = get_financial_news(company)
        if not articles:
            return "⚠ No recent news found."

        sentiments = []
        response_text = ""

        for article in articles:
            title = article["title"]
            sentiment, confidence = analyze_sentiment(title)
            sentiments.append(sentiment)

            response_text += f"\n📰 {title}\n"
            response_text += f"   Sentiment: {sentiment} (Confidence: {confidence})\n\n"

        overall = max(set(sentiments), key=sentiments.count)
        response_text += f"\n📈 Overall Market Sentiment: {overall.upper()}"

        return response_text

    return "Ask me about:\n- 'price AAPL'\n- 'news Tesla'"

# -----------------------------------
# CHAT MEMORY
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about stock price or financial news..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = generate_response(prompt)
            st.markdown(response)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
