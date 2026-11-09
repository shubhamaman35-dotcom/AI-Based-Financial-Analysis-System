"""
View: NLP Financial News & Sentiment Radar
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.ui.components import UIComponents
from src.data.news_fetcher import FinancialNewsFetcher
from src.nlp.sentiment_engine import SentimentEngine

def render_sentiment_view(ticker: str):
    st.markdown("### 📰 NLP Financial News & Market Sentiment Radar")

    news_fetcher = FinancialNewsFetcher()
    sentiment_engine = SentimentEngine()

    with st.spinner(f"Aggregating real-time financial headlines and computing sentiment scores for {ticker}..."):
        raw_news = news_fetcher.fetch_news(ticker, max_articles=12)
        sent_results = sentiment_engine.analyze_news_feed(raw_news)

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s_score = sent_results["overall_sentiment"]
        UIComponents.render_metric_card("Sentiment Polarity", f"{s_score}", f"{sent_results['overall_label']}", is_positive=s_score >= 0)
    with c2:
        UIComponents.render_metric_card("Bullish Headlines", f"{sent_results['bullish_pct']}%", is_positive=True)
    with c3:
        UIComponents.render_metric_card("Bearish Headlines", f"{sent_results['bearish_pct']}%", is_positive=False)
    with c4:
        UIComponents.render_metric_card("Neutral / Unbiased", f"{sent_results['neutral_pct']}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Charts Row
    ch1, ch2 = st.columns([1, 1])

    with ch1:
        # Donut Chart for Sentiment Distribution
        donut_fig = go.Figure(data=[go.Pie(
            labels=["Bullish", "Neutral", "Bearish"],
            values=[sent_results["bullish_pct"], sent_results["neutral_pct"], sent_results["bearish_pct"]],
            hole=0.55,
            marker=dict(colors=["#00FFA3", "#FFAA00", "#FF4B4B"])
        )])
        donut_layout = UIComponents.get_plotly_dark_layout("Headline Sentiment Distribution")
        donut_layout["height"] = 340
        donut_fig.update_layout(donut_layout)
        st.plotly_chart(donut_fig, use_container_width=True)

    with ch2:
        # Top Keywords Bar Chart
        keywords = sent_results.get("top_keywords", {})
        if keywords:
            kw_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Mentions"]).sort_values(by="Mentions", ascending=True)
            kw_fig = go.Figure(go.Bar(
                x=kw_df["Mentions"],
                y=kw_df["Keyword"],
                orientation="h",
                marker=dict(color="#00B4D8")
            ))
            kw_layout = UIComponents.get_plotly_dark_layout("Top Financial Themes & Extracted Entities")
            kw_layout["height"] = 340
            kw_fig.update_layout(kw_layout)
            st.plotly_chart(kw_fig, use_container_width=True)
        else:
            st.info("No prominent keyword clusters detected.")

    # News Feed List
    st.markdown("### 📡 Live Feed & Headline Sentiment Tagging")
    articles = sent_results.get("analyzed_articles", [])

    if not articles:
        st.info("No recent news articles found for this ticker.")
    else:
        for art in articles:
            s_info = art.get("sentiment", {})
            lbl = s_info.get("label", "NEUTRAL")
            color = s_info.get("color", "#FFAA00")
            comp = s_info.get("compound", 0.0)

            st.markdown(
                f"""
                <div style="background-color: #151922; border-left: 4px solid {color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; border: 1px solid #1E293B;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div>
                            <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">{art.get('publisher', 'Financial Wire')}</span>
                            <span style="color: #64748B; font-size: 0.8rem;"> &bull; {art.get('published_at', '')}</span>
                        </div>
                        <span class="badge-pill" style="background: {color}22; color: {color}; border: 1px solid {color}44;">
                            {lbl} ({comp:+.2f})
                        </span>
                    </div>
                    <div style="font-size: 0.98rem; font-weight: 600; color: #F1F5F9; margin-bottom: 6px;">
                        <a href="{art.get('link', '#')}" target="_blank" style="color: #F1F5F9; text-decoration: none;">
                            {art.get('title', '')}
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
