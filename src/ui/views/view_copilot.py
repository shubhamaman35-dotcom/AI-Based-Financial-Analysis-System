"""
View: AI Financial Copilot & PDF Report Generator
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.nlp.copilot_agent import FinancialCopilotAgent, PERSONA_SYSTEM_PROMPTS
from src.reports.pdf_generator import InstitutionalReportGenerator

def render_copilot_view(
    ticker: str,
    company_name: str,
    quote: dict,
    fundamentals: dict,
    verdict_data: dict,
    dl_forecast_data: dict,
    risk_data: dict,
    sentiment_data: dict
):
    st.markdown("### 🤖 AI Financial Copilot & Institutional Report Generator")

    copilot = FinancialCopilotAgent()
    report_gen = InstitutionalReportGenerator()

    # Persona Selection Row
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        persona = st.selectbox("Select AI Copilot Persona", list(PERSONA_SYSTEM_PROMPTS.keys()), index=0)
    with col_p2:
        st.info(f"**Current Persona Objective:** {PERSONA_SYSTEM_PROMPTS[persona]}")

    st.markdown("---")

    # 1-Click Report and Thesis Generation Row
    c_btn1, c_btn2 = st.columns([1, 1])

    with c_btn1:
        if st.button("📝 Generate Institutional Investment Memo", use_container_width=True):
            thesis_text = copilot.generate_investment_thesis(
                ticker=ticker,
                company_name=company_name,
                quote_summary=quote,
                fundamentals=fundamentals,
                ml_verdict=verdict_data,
                sentiment_summary=sentiment_data,
                persona=persona
            )
            st.session_state["latest_thesis"] = thesis_text

    with c_btn2:
        if st.button("📥 Generate & Compile Institutional Research Report (PDF)", use_container_width=True):
            with st.spinner("Compiling quantitative analytics, charts, and risk profile into PDF..."):
                pdf_path = report_gen.generate_pdf_report(
                    ticker=ticker,
                    company_name=company_name,
                    quote=quote,
                    fundamentals=fundamentals,
                    verdict_data=verdict_data,
                    dl_forecast_data=dl_forecast_data,
                    risk_data=risk_data,
                    sentiment_data=sentiment_data
                )
                st.session_state["generated_pdf_path"] = str(pdf_path)

    # Show PDF Download Button if generated
    if "generated_pdf_path" in st.session_state and Path(st.session_state["generated_pdf_path"]).exists():
        pdf_file = Path(st.session_state["generated_pdf_path"])
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        mime_type = "application/pdf" if pdf_file.suffix == ".pdf" else "text/markdown"
        st.success(f"Report compiled successfully: `{pdf_file.name}`")
        st.download_button(
            label=f"⬇️ Download {pdf_file.name}",
            data=pdf_bytes,
            file_name=pdf_file.name,
            mime=mime_type,
            use_container_width=True
        )

    # Render Latest Thesis Memo
    if "latest_thesis" in st.session_state:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #151922; border: 1px solid #334155; border-radius: 10px; padding: 22px; margin-bottom: 20px;">
            """,
            unsafe_allow_html=True
        )
        st.markdown(st.session_state["latest_thesis"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Conversational Assistant Section
    st.markdown("### 💬 Interactive Market Intelligence Inquiry")
    st.caption("Ask specific questions regarding valuation, risk buffers, catalysts, or AI model rationale:")

    # Initialize chat history
    if "copilot_chat_history" not in st.session_state:
        st.session_state["copilot_chat_history"] = [
            {"role": "assistant", "content": f"Hello! I am your AI Financial Copilot configured as **{persona}**. Ask me any question about {ticker} valuation, catalysts, downside variance, or model forecasts."}
        ]

    for msg in st.session_state["copilot_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(f"Ask AI Copilot about {ticker}...")
    if user_prompt:
        st.session_state["copilot_chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        context_data = {
            "current_price": quote.get("current_price", 150.0),
            "verdict": verdict_data.get("verdict", "BUY"),
            "health_score": fundamentals.get("health_score", 80),
            "sentiment_label": sentiment_data.get("overall_label", "BULLISH")
        }

        with st.spinner("AI Copilot analyzing financial factors..."):
            reply = copilot.answer_query(
                user_query=user_prompt,
                ticker=ticker,
                context_data=context_data,
                persona=persona
            )

        st.session_state["copilot_chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
