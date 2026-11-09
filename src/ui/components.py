"""
UI Components and Visual Styling Module
Provides modern dark-mode financial terminal styling, custom CSS, metric cards, and chart formatting.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

class UIComponents:
    """
    Renders styled HTML cards, headers, badges, and Plotly templates.
    """

    @staticmethod
    def apply_custom_css():
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }

            code, pre, .font-mono {
                font-family: 'JetBrains Mono', monospace !important;
            }

            .main {
                background-color: #0B0E14;
            }

            /* Fix Streamlit Top Container Clipping */
            .block-container {
                padding-top: 2.5rem !important;
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
                padding-bottom: 2rem !important;
            }

            header[data-testid="stHeader"] {
                background: transparent !important;
                height: 1.5rem !important;
                z-index: 1 !important;
            }

            /* Top Header Ticker Bar */
            #top-bar-header {
                height: 50px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                overflow: visible !important;
                background-color: #1e222d !important;
                border: 1px solid #2a2e39 !important;
                border-radius: 8px !important;
                padding: 0 18px !important;
                margin-bottom: 16px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
            }

            #top-bar-header * {
                vertical-align: middle !important;
            }

            /* High-Contrast Watchlist & Action Buttons on Dark Backgrounds */
            div.stButton > button {
                color: #FFFFFF !important;
                background-color: #2a2e39 !important;
                border: 1px solid #3a3f50 !important;
                border-radius: 6px !important;
                font-weight: 600 !important;
                transition: all 0.15s ease !important;
            }

            div.stButton > button:hover {
                background-color: #00FFA3 !important;
                color: #0B0E14 !important;
                border-color: #00FFA3 !important;
                transform: translateY(-1px) !important;
            }

            div.stButton > button:active {
                background-color: #00B4D8 !important;
                color: #0B0E14 !important;
            }

            /* Custom metric container card */
            .fin-card {
                background: linear-gradient(135deg, #151922 0%, #1A202C 100%);
                border: 1px solid #2D3748;
                border-radius: 12px;
                padding: 18px 22px;
                margin-bottom: 16px;
                box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.35);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .fin-card:hover {
                border-color: #00FFA3;
                transform: translateY(-2px);
            }

            .card-title {
                font-size: 0.85rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #94A3B8;
                margin-bottom: 6px;
            }

            .card-value {
                font-size: 1.65rem;
                font-weight: 700;
                color: #FFFFFF;
                font-family: 'JetBrains Mono', monospace;
            }

            .card-delta {
                font-size: 0.9rem;
                font-weight: 600;
                margin-top: 4px;
            }

            .delta-pos {
                color: #00FFA3;
            }
            .delta-neg {
                color: #FF4B4B;
            }

            /* Badge pill */
            .badge-pill {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            /* Verdict highlight hero card */
            .verdict-hero {
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                border: 2px solid #3B82F6;
                border-radius: 14px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.2);
            }

            /* Streamlit tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: #11151F;
                padding: 8px;
                border-radius: 12px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 44px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 8px;
                color: #94A3B8;
                font-size: 0.9rem;
                font-weight: 600;
                padding: 0 16px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1E293B !important;
                color: #00FFA3 !important;
                border: 1px solid #334155;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_header():
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 14px; margin-bottom: 20px;">
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(90deg, #00FFA3, #00B4D8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        ⚡ QUANTUM INTELLIGENCE TERMINAL
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #94A3B8;">
                        AI-Based Multi-Horizon Financial Analysis & Quantitative Investment Intelligence System (CSAIML)
                    </p>
                </div>
                <div style="text-align: right;">
                    <span class="badge-pill" style="background: rgba(0, 255, 163, 0.15); color: #00FFA3; border: 1px solid rgba(0, 255, 163, 0.3);">
                        ● LIVE AI ENGINE ONLINE
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_metric_card(title: str, value: str, delta: str = None, is_positive: bool = True):
        delta_html = ""
        if delta:
            css_class = "delta-pos" if is_positive else "delta-neg"
            arrow = "▲" if is_positive else "▼"
            delta_html = f'<div class="card-delta {css_class}">{arrow} {delta}</div>'

        st.markdown(
            f"""
            <div class="fin-card">
                <div class="card-title">{title}</div>
                <div class="card-value">{value}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_verdict_hero(verdict_data: Dict[str, Any], ticker: str, current_price: float):
        verdict = verdict_data.get("verdict", "BUY")
        score = verdict_data.get("composite_score", 75.0)
        action = verdict_data.get("action_code", "ACCUMULATE")
        desc = verdict_data.get("summary_description", "")
        v_color = verdict_data.get("verdict_color", "#00FFA3")

        st.markdown(
            f"""
            <div class="verdict-hero" style="border-color: {v_color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span class="badge-pill" style="background: {v_color}22; color: {v_color}; border: 1px solid {v_color}55;">
                            PRIMARY AI RATING: {verdict}
                        </span>
                        <h2 style="margin: 10px 0 4px 0; font-size: 2rem; font-weight: 800; color: #FFFFFF;">
                            {ticker} &bull; ${current_price}
                        </h2>
                        <p style="margin: 0; font-size: 0.95rem; color: #CBD5E1; max-width: 700px; line-height: 1.5;">
                            {desc}
                        </p>
                    </div>
                    <div style="text-align: right; background: #0B0E14; padding: 14px 20px; border-radius: 10px; border: 1px solid #334155;">
                        <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">AI Consensus Score</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {v_color}; font-family: 'JetBrains Mono', monospace;">
                            {score}<span style="font-size: 1rem; color: #64748B;">/100</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">Action: <strong style="color: #FFFFFF;">{action}</strong></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def get_plotly_dark_layout(title: str = "") -> Dict[str, Any]:
        return {
            "title": {"text": title, "font": {"family": "Inter", "size": 16, "color": "#FFFFFF"}},
            "paper_bgcolor": "#11151F",
            "plot_bgcolor": "#11151F",
            "font": {"color": "#94A3B8", "family": "Inter"},
            "xaxis": {
                "gridcolor": "#1E293B",
                "zerolinecolor": "#334155",
                "showline": True,
                "linecolor": "#334155"
            },
            "yaxis": {
                "gridcolor": "#1E293B",
                "zerolinecolor": "#334155",
                "showline": True,
                "linecolor": "#334155"
            },
            "margin": {"l": 50, "r": 30, "t": 45, "b": 40},
            "legend": {"orientation": "h", "y": 1.05, "x": 0.0, "font": {"size": 11}}
        }
