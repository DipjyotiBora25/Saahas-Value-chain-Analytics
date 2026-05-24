"""Floating Groq-powered chatbot for interpreting uploaded sales & purchase data.

Renders a circular robot button fixed to the bottom-right corner. The button
pulses and shows a "Need help?" nudge after the user has been idle for 60s.
Clicking opens a popover with a chat interface that uses Groq Cloud for LLM
inference.

Setup:
    Set GROQ_API_KEY and GROQ_MODEL_NAME in `.env` or your deployment environment.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
IDLE_MS = 60_000
MAX_HISTORY_TURNS = 8

@st.cache_resource
def get_groq_client() -> Groq:
    """Initialize and cache the Groq client using environment configuration."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is missing. Set it in .env or in your deployment environment."
        )
    return Groq(api_key=api_key)


def get_chat_response(messages: list[dict], model_name: str | None = None) -> str:
    """Call the Groq Cloud chat completions API and return the assistant response."""
    model_name = model_name or os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_DEFAULT)
    client = get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )
        if hasattr(completion, "choices") and completion.choices:
            return completion.choices[0].message.content
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Groq Cloud API: {e}")
        raise


def _safe_popover(label: str, help: str | None = None):
    if hasattr(st, "popover"):
        return st.popover(label, help=help)
    return st.expander(label, expanded=False)


def _safe_toggle(label: str, value: bool = False, key: str | None = None) -> bool:
    if hasattr(st, "toggle"):
        return st.toggle(label, value=value, key=key)
    return st.checkbox(label, value=value, key=key)

# History compaction: keep the last N user/assistant turns to avoid
# re-evaluating the entire transcript every request.
MAX_HISTORY_TURNS = 8

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _format_reasoning(text: str) -> str:
    """Collapse <think>…</think> blocks into expandable HTML <details> so the
    final answer is visible up front but the reasoning trail is still
    inspectable. Trailing unterminated <think> tags (mid-stream) become an
    open 'thinking…' marker."""
    def repl(m: re.Match) -> str:
        body = m.group(1).strip()
        return (
            "<details style='margin:0.25rem 0;color:#64748b;'>"
            "<summary style='cursor:pointer;'>💭 Reasoning</summary>\n\n"
            f"{body}\n\n</details>\n\n"
        )

    out = _THINK_RE.sub(repl, text)
    if "<think>" in out and "</think>" not in out:
        out = out.replace("<think>", "_💭 thinking…_\n\n", 1)
    return out


# ---------------------------------------------------------------------------
# Data context — turns the uploaded dataframes into a compact text summary


# ---------------------------------------------------------------------------
# Data context — turns the uploaded dataframes into a compact text summary
# ---------------------------------------------------------------------------

def _amount_col(df: pd.DataFrame, prefer: list[str]) -> str | None:
    for c in prefer:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "total" in cl or "amount" in cl:
            return c
    return None


def _top_summary(df: pd.DataFrame, group_col: str, value_col: str, n: int = 5) -> str:
    if group_col not in df.columns or value_col not in df.columns:
        return ""
    agg = (
        df.groupby(group_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    if agg.empty:
        return ""
    return "; ".join(f"{idx}: ₹{val:,.0f}" for idx, val in agg.items())


def build_data_context(sales_df: pd.DataFrame, purchase_df: pd.DataFrame) -> str:
    parts: list[str] = []

    if sales_df is None or sales_df.empty:
        parts.append("SALES DATA: none uploaded.")
    else:
        sa = _amount_col(sales_df, ["Item Total"])
        rows = len(sales_df)
        total_rev = float(pd.to_numeric(sales_df[sa], errors="coerce").sum()) if sa else 0.0
        cust_n = sales_df["Customer Name"].nunique() if "Customer Name" in sales_df.columns else 0
        inv_n = sales_df["Invoice Number"].nunique() if "Invoice Number" in sales_df.columns else rows
        date_range = ""
        if "Invoice Date" in sales_df.columns:
            d = pd.to_datetime(sales_df["Invoice Date"], errors="coerce").dropna()
            if not d.empty:
                date_range = f" Date range: {d.min().date()} to {d.max().date()}."
        parts.append(
            f"SALES DATA: {rows:,} rows, {inv_n:,} invoices, {cust_n:,} unique customers. "
            f"Total revenue: ₹{total_rev:,.0f}.{date_range}"
        )
        parts.append(f"Sales columns: {', '.join(map(str, sales_df.columns))}")
        if sa:
            top_cust = _top_summary(sales_df, "Customer Name", sa, 5)
            if top_cust:
                parts.append(f"Top 5 customers by revenue: {top_cust}")
            top_item = _top_summary(sales_df, "Item Name", sa, 5)
            if top_item:
                parts.append(f"Top 5 selling items: {top_item}")
            top_vert = _top_summary(sales_df, "CF.Business Verticals", sa, 5)
            if top_vert:
                parts.append(f"Revenue by business vertical: {top_vert}")
            if "YearMonth" in sales_df.columns:
                monthly = _top_summary(sales_df, "YearMonth", sa, 12)
                if monthly:
                    parts.append(f"Monthly revenue: {monthly}")

    if purchase_df is None or purchase_df.empty:
        parts.append("PURCHASE DATA: none uploaded.")
    else:
        pa = _amount_col(purchase_df, ["Sum of Total Amount", "Total Amount"])
        rows = len(purchase_df)
        total_sp = float(pd.to_numeric(purchase_df[pa], errors="coerce").sum()) if pa else 0.0
        ven_n = purchase_df["Vendor Name"].nunique() if "Vendor Name" in purchase_df.columns else 0
        inv_n = purchase_df["Invoice #"].nunique() if "Invoice #" in purchase_df.columns else rows
        date_range = ""
        if "Invoice Date" in purchase_df.columns:
            d = pd.to_datetime(purchase_df["Invoice Date"], errors="coerce").dropna()
            if not d.empty:
                date_range = f" Date range: {d.min().date()} to {d.max().date()}."
        parts.append(
            f"PURCHASE DATA: {rows:,} rows, {inv_n:,} invoices, {ven_n:,} unique vendors. "
            f"Total spend: ₹{total_sp:,.0f}.{date_range}"
        )
        parts.append(f"Purchase columns: {', '.join(map(str, purchase_df.columns))}")
        if pa:
            top_ven = _top_summary(purchase_df, "Vendor Name", pa, 5)
            if top_ven:
                parts.append(f"Top 5 vendors by spend: {top_ven}")
            top_cat = _top_summary(purchase_df, "Category", pa, 5)
            if top_cat:
                parts.append(f"Top categories by spend: {top_cat}")
            top_vert = _top_summary(purchase_df, "Business Vertical", pa, 5)
            if top_vert:
                parts.append(f"Spend by business vertical: {top_vert}")
            if "YearMonth" in purchase_df.columns:
                monthly = _top_summary(purchase_df, "YearMonth", pa, 12)
                if monthly:
                    parts.append(f"Monthly spend: {monthly}")

    if (sales_df is not None and not sales_df.empty) and (
        purchase_df is not None and not purchase_df.empty
    ):
        sa = _amount_col(sales_df, ["Item Total"])
        pa = _amount_col(purchase_df, ["Sum of Total Amount", "Total Amount"])
        if sa and pa:
            rev = float(pd.to_numeric(sales_df[sa], errors="coerce").sum())
            spd = float(pd.to_numeric(purchase_df[pa], errors="coerce").sum())
            profit = rev - spd
            margin = (profit / rev * 100) if rev else 0
            parts.append(
                f"COMBINED: gross profit ₹{profit:,.0f} ({margin:.1f}% margin on revenue)."
            )

    return "\n".join(parts)


def _df_fingerprint(df: pd.DataFrame | None) -> tuple | None:
    """Cheap signature: shape + columns + a sample row. Used to detect when
    the data context needs to be rebuilt — otherwise we keep the identical
    assistant prompt across turns for more consistent Groq responses."""
    if df is None or df.empty:
        return None
    try:
        sample = tuple(df.iloc[0].astype(str).tolist())
    except Exception:
        sample = ()
    return (df.shape, tuple(map(str, df.columns)), sample)


def get_data_context_cached(sales_df: pd.DataFrame, purchase_df: pd.DataFrame) -> str:
    fp = (_df_fingerprint(sales_df), _df_fingerprint(purchase_df))
    if st.session_state.get("_ctx_fp") == fp and "_ctx" in st.session_state:
        return st.session_state["_ctx"]
    ctx = build_data_context(sales_df, purchase_df)
    st.session_state["_ctx_fp"] = fp
    st.session_state["_ctx"] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Floating UI: CSS to pin the popover bottom-right + JS to detect idle
# ---------------------------------------------------------------------------

_FLOATING_CSS = """
<style>
    /* Pin the LAST popover on the page (our robot) to the bottom-right corner */
    div[data-testid="stPopover"]:last-of-type {
        position: fixed !important;
        bottom: calc(24px + 1cm);
        right: 24px;
        z-index: 9999;
        width: auto !important;
    }
    /* Style the trigger button as a floating circular paperclip (Clippy vibe) */
    div[data-testid="stPopover"]:last-of-type > div > button {
        width: 120px !important;
        height: 120px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #fde68a 0%, #f59e0b 60%, #d97706 100%) !important;
        color: #1e293b !important;
        font-size: 64px !important;
        line-height: 1 !important;
        border: 4px solid #ffffff !important;
        box-shadow: 0 18px 42px rgba(217,119,6,0.55) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s ease;
    }
    div[data-testid="stPopover"]:last-of-type > div > button:hover {
        transform: scale(1.08) rotate(-6deg);
    }
    /* Idle: pulse the button and show a speech-bubble nudge */
    body.szw-idle div[data-testid="stPopover"]:last-of-type > div > button {
        animation: szw-pulse 1.8s ease-in-out infinite;
    }
    body.szw-idle div[data-testid="stPopover"]:last-of-type::before {
        content: "Need help with your data?";
        position: absolute;
        bottom: 132px;
        right: 0;
        background: #ffffff;
        padding: 10px 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.18);
        font-size: 13px;
        font-weight: 600;
        color: #0f172a;
        white-space: nowrap;
        animation: szw-float 2.4s ease-in-out infinite;
    }
    body.szw-idle div[data-testid="stPopover"]:last-of-type::after {
        content: "";
        position: absolute;
        bottom: 122px;
        right: 48px;
        width: 12px;
        height: 12px;
        background: #ffffff;
        transform: rotate(45deg);
        box-shadow: 3px 3px 6px rgba(15,23,42,0.08);
    }
    @keyframes szw-pulse {
        0%, 100% { box-shadow: 0 18px 42px rgba(217,119,6,0.55), 0 0 0 0 rgba(245,158,11,0.75); }
        50%      { box-shadow: 0 18px 42px rgba(217,119,6,0.55), 0 0 0 28px rgba(245,158,11,0); }
    }
    @keyframes szw-float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-4px); }
    }
    /* Tighten the popover panel */
    div[data-testid="stPopoverBody"] {
        min-width: 380px;
        max-width: 440px;
    }
</style>
"""

_IDLE_JS = f"""
<script>
(function() {{
    const parentWin = window.parent;
    if (parentWin.__szwIdleSetup) return;
    parentWin.__szwIdleSetup = true;

    const doc = parentWin.document;
    const body = doc.body;
    parentWin.__szwLastActivity = Date.now();

    const reset = () => {{
        parentWin.__szwLastActivity = Date.now();
        body.classList.remove('szw-idle');
    }};
    ['mousemove', 'keydown', 'scroll', 'click', 'touchstart'].forEach(ev =>
        doc.addEventListener(ev, reset, {{ passive: true, capture: true }})
    );

    setInterval(() => {{
        if (Date.now() - parentWin.__szwLastActivity > {IDLE_MS}) {{
            body.classList.add('szw-idle');
        }}
    }}, 4000);
}})();
</script>
"""


def _inject_floating_assets() -> None:
    st.markdown(_FLOATING_CSS, unsafe_allow_html=True)
    components.html(_IDLE_JS, height=0)


# ---------------------------------------------------------------------------
# Chat handling — explicit accumulator so streaming is reliable
# ---------------------------------------------------------------------------

def _process_query(prompt: str, model: str, data_context: str) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are a data analyst embedded in the Saahas Zero Waste Analytics dashboard. "
            "Answer the user's question using ONLY the DATA CONTEXT below. "
            "Quote concrete numbers in ₹ where helpful, prefer short bullet points, "
            "and if the answer is not in the context, say so plainly and suggest which "
            "dashboard tab to look at (Overview, Sales, Purchase, Revenue Insights, Spend Analysis).\n\n"
            f"=== DATA CONTEXT ===\n{data_context}\n=== END CONTEXT ==="
        ),
    }
    history = st.session_state["chat_messages"][-(MAX_HISTORY_TURNS * 2):]
    api_messages = [system_msg] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]

    placeholder = st.empty()
    placeholder.markdown("_Thinking…_")
    try:
        reply = get_chat_response(api_messages, model_name=model)
        if not reply:
            msg = "_(empty response from model — try a different prompt)_"
            placeholder.markdown(msg)
            return msg
        placeholder.markdown(_format_reasoning(reply), unsafe_allow_html=True)
        return reply
    except ValueError as e:
        msg = str(e)
        placeholder.error(msg)
        return msg
    except Exception as e:
        msg = f"❌ Groq Cloud API error: {e}"
        placeholder.error(msg)
        return msg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_floating_chatbot(sales_df: pd.DataFrame, purchase_df: pd.DataFrame) -> None:
    _inject_floating_assets()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "groq_model" not in st.session_state:
        st.session_state["groq_model"] = os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_DEFAULT)

    data_context = get_data_context_cached(sales_df, purchase_df)
    no_data = (sales_df is None or sales_df.empty) and (
        purchase_df is None or purchase_df.empty
    )

    with _safe_popover("📎", help="Ask the data assistant"):
        st.markdown("### 📎 Saahas Data Assistant")
        st.caption("Runs via Groq Cloud API. Configure GROQ_API_KEY in .env or deployment secrets.")

        with st.expander("⚙️ Settings", expanded=False):
            model = st.text_input(
                "Groq model name",
                value=st.session_state.get("groq_model", os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_DEFAULT)),
                key="groq_model_input",
            )
            st.session_state["groq_model"] = model
            if not os.getenv("GROQ_API_KEY"):
                st.error(
                    "GROQ_API_KEY is not configured. Set GROQ_API_KEY in `.env` or your deployment environment."
                )
            st.markdown(
                "This assistant uses Groq Cloud for inference. "
                "Configure `GROQ_API_KEY` and optionally `GROQ_MODEL_NAME` in your environment."
            )

            show_ctx = _safe_toggle(
                "Show data context",
                value=st.session_state.get("show_ctx", False),
                key="show_ctx",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear chat", use_container_width=True):
                    st.session_state["chat_messages"] = []
                    st.rerun()
            with col_b:
                if st.button("Refresh", use_container_width=True):
                    st.experimental_rerun()

        if no_data:
            st.info("📂 Upload sales or purchase data above to chat about it.")
        elif st.session_state.get("show_ctx"):
            with st.expander("Data context sent to model", expanded=False):
                st.code(data_context, language="text")

        # Suggested prompts as quick buttons
        if not st.session_state["chat_messages"] and not no_data:
            st.markdown("**Try asking:**")
            suggestions = [
                "Summarize the key revenue and spend numbers.",
                "Which vendors should I review first?",
                "What's driving profit or loss?",
                "Top 3 categories by spend?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                if cols[i % 2].button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["_pending_prompt"] = s

        # Render history
        msg_box = st.container(height=320)
        with msg_box:
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.markdown(
                            _format_reasoning(msg["content"]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(msg["content"])

        # Input form (more reliable inside a popover than st.chat_input)
        prompt = st.session_state.pop("_pending_prompt", None)
        with st.form(key="chatbot_form", clear_on_submit=True):
            typed = st.text_input(
                "Question",
                placeholder="Ask about revenue, vendors, categories, trends…",
                label_visibility="collapsed",
                key="chatbot_input",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)
        if submitted and typed and typed.strip():
            prompt = typed.strip()

        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with msg_box:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    reply = _process_query(
                        prompt,
                        st.session_state["groq_model"],
                        data_context,
                    )
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})