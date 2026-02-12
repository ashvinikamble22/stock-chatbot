import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

# -----------------------------------
# CONFIG
# -----------------------------------

st.set_page_config(page_title="Financial AI Chatbot", layout="wide")
st.title("📊 Financial News Sentiment Chatbot (FinBERT)")

# -----------------------------------
# ENTER YOUR NEWS API KEY HERE
# -----------------------------------

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# -----------------------------------
# LOAD FINBERT (Cached for performance)
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
# SENTIMENT FUNCTION
# -----------------------------------

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    
    sentiment_id = torch.argmax(probs).item()
    confidence = probs[0][sentiment_id].item()
    
    return labels[sentiment_id], round(confidence, 3)

# -----------------------------------
# GET STOCK PRICE
# -----------------------------------

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol.upper())
    data = ticker.history(period="1d")
    
    if data.empty:
        return None
    
    return round(data["Close"].iloc[-1], 2)

# -----------------------------------
# GET NEWS
# -----------------------------------

def get_financial_news(company):
    articles = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        page_size=5
    )
    
    return articles["articles"]

# -----------------------------------
# USER INPUT
# -----------------------------------

user_input = st.text_input("Ask something (Example: price AAPL or news Tesla)")

if user_input:

    # -------------------------------
    # STOCK PRICE QUERY
    # -------------------------------
    if "price" in user_input.lower():
        symbol = user_input.split()[-1]
        price = get_stock_price(symbol)

        if price:
            st.success(f"💰 Current price of {symbol.upper()} is ${price}")
        else:
            st.error("Stock symbol not found.")

    # -------------------------------
    # NEWS + SENTIMENT QUERY
    # -------------------------------
    elif "news" in user_input.lower():
        company = user_input.replace("news", "").strip()

        with st.spinner("Fetching news and analyzing sentiment..."):
            articles = get_financial_news(company)

            if not articles:
                st.warning("No news found.")
            else:
                results = []
                sentiments = []

                for article in articles:
                    title = article["title"]
                    sentiment, confidence = analyze_sentiment(title)

                    sentiments.append(sentiment)

                    results.append({
                        "Title": title,
                        "Sentiment": sentiment,
                        "Confidence": confidence
                    })

                df = pd.DataFrame(results)

                # Overall Sentiment
                overall = max(set(sentiments), key=sentiments.count)

                st.subheader(f"📈 Overall Sentiment: {overall.upper()}")

                st.dataframe(df, use_container_width=True)

    else:
        st.info("Please ask about stock price or financial news sentiment.")
