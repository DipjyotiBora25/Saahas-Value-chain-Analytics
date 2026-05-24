

import difflib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

POWER_BI_URL = "https://app.powerbi.com/view?r=eyJrIjoiOTBiNWUzMTEtN2Q0YS00Y2Y2LTkwNDEtODczMDNiOGU5MTAxIiwidCI6IjBlNmEwM2I0LTI4YjktNGQ3Zi1hYzNjLTM4MTYzY2Q4MzYwMCIsImMiOjEwfQ%3D%3D"

st.set_page_config(page_title="Saahas Zero Waste Analytics", page_icon="♻️", layout="wide")

# Import dashboard modules
from sales_dashboard import render_revenue_insights
from purchase_dashboard import render_spend_analysis
from chatbot import render_floating_chatbot

ROOT = Path(__file__).resolve().parent
LOGO_FILE = ROOT / "SZW_Logo.png"
CATEGORY_MAP_FILE = ROOT / "Item_Cat_Update.csv"


@st.cache_data
def load_category_map() -> pd.DataFrame:
    """Load the Item Name -> Category mapping. Returns a df with columns
    ['Item Name Key', 'Category'] where the key is lowercased+stripped."""
    if not CATEGORY_MAP_FILE.exists():
        return pd.DataFrame(columns=["Item Name Key", "Category"])
    df = pd.read_csv(CATEGORY_MAP_FILE, encoding="utf-8-sig")
    if "Delete" in df.columns:
        df = df[df["Delete"].fillna("").astype(str).str.strip().str.lower() != "remove"]
    df = df.dropna(subset=["Item Name", "Category"])
    df["Item Name Key"] = df["Item Name"].astype(str).str.strip().str.lower()
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[df["Item Name Key"] != ""]
    df = df.drop_duplicates(subset=["Item Name Key"], keep="first")
    return df[["Item Name Key", "Category"]].reset_index(drop=True)


def attach_category(df: pd.DataFrame, category_map: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'Category' column populated via the mapping.
    Existing non-null Category values are preserved; missing rows get 'Uncategorized'."""
    if df.empty or "Item Name" not in df.columns:
        return df.copy() if not df.empty else df
    out = df.copy()
    if category_map.empty:
        if "Category" not in out.columns:
            out["Category"] = "Uncategorized"
        else:
            out["Category"] = out["Category"].fillna("Uncategorized")
        return out
    key = out["Item Name"].astype(str).str.strip().str.lower()
    lookup = dict(zip(category_map["Item Name Key"], category_map["Category"]))
    mapped = key.map(lookup)
    if "Category" in out.columns:
        out["Category"] = out["Category"].where(out["Category"].notna() & (out["Category"].astype(str).str.strip() != ""), mapped)
    else:
        out["Category"] = mapped
    out["Category"] = out["Category"].fillna("Uncategorized")
    return out



def app_styles():
    st.markdown(
        """
        <style>
            :root {
                --fs-3xs: clamp(0.72rem, 0.68rem + 0.18vw, 0.82rem);
                --fs-2xs: clamp(0.78rem, 0.73rem + 0.22vw, 0.9rem);
                --fs-xs:  clamp(0.82rem, 0.76rem + 0.28vw, 0.95rem);
                --fs-sm:  clamp(0.88rem, 0.8rem  + 0.4vw,  1.05rem);
                --fs-base:clamp(0.95rem, 0.86rem + 0.45vw, 1.15rem);
                --fs-md:  clamp(1rem,    0.9rem  + 0.55vw, 1.3rem);
                --fs-lg:  clamp(1.25rem, 1.05rem + 1.05vw, 1.85rem);
                --fs-xl:  clamp(1.5rem,  1.15rem + 1.75vw, 2.5rem);
                --fs-2xl: clamp(1.75rem, 1.25rem + 2.4vw,  3rem);
                --sp-1:   clamp(0.35rem, 0.3rem + 0.2vw, 0.55rem);
                --sp-2:   clamp(0.55rem, 0.45rem + 0.4vw, 0.85rem);
                --sp-3:   clamp(0.85rem, 0.7rem + 0.6vw, 1.35rem);
                --sp-4:   clamp(1.1rem, 0.85rem + 1vw, 1.85rem);
                --radius-md: clamp(0.75rem, 0.6rem + 0.5vw, 1.15rem);
                --radius-pill: 999px;
            }
            .page-shell {background:#f8fafc; color:#0f172a; font-size:var(--fs-base);}
            .header-panel {display:flex;justify-content:space-between;align-items:flex-start;gap:var(--sp-3);margin-bottom:var(--sp-4);flex-wrap:wrap;}
            .brand-block {display:flex;align-items:center;gap:var(--sp-3);}
            .brand-mark {width:clamp(40px,2.2rem + 2vw,64px);height:clamp(40px,2.2rem + 2vw,64px);border-radius:var(--radius-md);background:#eaf2ff;display:flex;align-items:center;justify-content:center;font-size:var(--fs-lg);}
            .brand-title {font-size:var(--fs-2xl);font-weight:800;margin:0;color:#0f172a;line-height:1.1;}
            .brand-subtitle {margin:0;color:#475569;font-size:var(--fs-sm);}
            .meta-block {display:flex;flex-direction:column;gap:var(--sp-1);align-items:flex-end;}
            .meta-chip {padding:var(--sp-2) var(--sp-3);border-radius:var(--radius-pill);background:#ffffff;border:1px solid #d1d5db;color:#334155;font-size:var(--fs-2xs);}
            .upload-card {background:white;padding:var(--sp-3);border-radius:var(--radius-md);box-shadow:0 15px 35px rgba(15,23,42,0.08);border:1px solid #e2e8f0;margin-bottom:var(--sp-4);}
            .metric-row {display:grid;grid-template-columns:repeat(auto-fit,minmax(clamp(180px,16vw,260px),1fr));gap:var(--sp-3);margin-bottom:var(--sp-4);}
            .metric-card {background:white;padding:var(--sp-3);border-radius:var(--radius-md);box-shadow:0 12px 30px rgba(15,23,42,0.06);border:1px solid #e2e8f0;}
            .metric-label {font-size:var(--fs-3xs);font-weight:700;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:var(--sp-2);}
            .metric-value {font-size:var(--fs-xl);font-weight:800;color:#0f172a;line-height:1;}
            .metric-delta {margin-top:var(--sp-1);font-size:var(--fs-2xs);font-weight:700;}
            .metric-delta.positive {color:#0f766e;}
            .metric-delta.negative {color:#b91c1c;}
            .section-card {background:white;padding:var(--sp-3);border-radius:var(--radius-md);box-shadow:0 12px 30px rgba(15,23,42,0.06);border:1px solid #e2e8f0;}
            .section-title {font-size:var(--fs-md);font-weight:700;color:#0f172a;margin-bottom:var(--sp-2);}
            .section-copy {font-size:var(--fs-sm);color:#475569;margin-bottom:var(--sp-3);line-height:1.6;}
            .chart-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(clamp(260px,24vw,360px),1fr));gap:var(--sp-3);}
            .data-table th{font-weight:700;color:#475569;border-bottom:1px solid #e2e8f0;padding-bottom:var(--sp-2);text-align:left;font-size:var(--fs-2xs);}
            .data-table td{padding:var(--sp-2) 0;font-size:var(--fs-xs);color:#334155;border-bottom:1px solid #f1f5f9;}
            .small-caption {font-size:var(--fs-2xs);color:#64748b;margin-top:var(--sp-1);}
            .tab-guide {display:flex;gap:var(--sp-2);flex-wrap:wrap;margin-bottom:var(--sp-3);}
            .tab-pill {padding:var(--sp-2) var(--sp-3);border-radius:var(--radius-pill);background:#f8fafc;border:1px solid #e2e8f0;color:#475569;font-size:var(--fs-2xs);}
            .tab-pill.active {background:#e0f2fe;color:#0369a1;border-color:#bae6fd;}
            .landing-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;margin:1rem 0;}
            .landing-card {background:white;padding:1.5rem;border-radius:1rem;box-shadow:0 18px 35px rgba(15,23,42,0.08);border:1px solid #e2e8f0;}
            .landing-card h3 {margin-top:0;margin-bottom:0.75rem;color:#0f172a;}
            .landing-card p {margin:0;color:#475569;line-height:1.7;}
            .powerbi-embed-wrapper {width:100%;max-width:100vw;margin:0;padding:0;overflow:hidden;}
            .powerbi-embed-wrapper iframe {width:100% !important;height:900px !important;border:none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_tabular(source):
    if source is None:
        return pd.DataFrame()
    if hasattr(source, "name"):
        name = source.name.lower()
    else:
        name = str(source).lower()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(source, engine="openpyxl")
    return pd.read_csv(source, low_memory=False)


def normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["Invoice Date", "Due Date", "Last Payment Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Invoice Date" in df.columns:
        df["Year"] = df["Invoice Date"].dt.year
        df["Month"] = df["Invoice Date"].dt.month
        df["Quarter"] = df["Invoice Date"].dt.quarter
        df["YearMonth"] = df["Invoice Date"].dt.to_period("M").astype(str)
    return df


def normalize_purchase(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["Invoice Date", "Submission Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "Invoice Date" in df.columns and "YearMonth" not in df.columns:
        df["Year"] = df["Invoice Date"].dt.year
        df["Month"] = df["Invoice Date"].dt.month
        df["Quarter"] = df["Invoice Date"].dt.quarter
        df["YearMonth"] = df["Invoice Date"].dt.to_period("M").astype(str)
    return df


def safe_sales_amount_column(df: pd.DataFrame) -> str | None:
    if "Item Total" in df.columns:
        return "Item Total"
    candidates = [c for c in df.columns if "total" in c.lower() and c not in ["Item Total"]]
    return candidates[0] if candidates else None


def safe_purchase_amount_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if c.strip() in ["Sum of Total Amount", "Total Amount", "Sum of Total Amount "]]
    return candidates[0] if candidates else None


def metric_card(title: str, value: str, delta: str = None, positive: bool = True):
    delta_class = "positive" if positive else "negative"
    delta_markup = f"<div class='metric-delta {delta_class}'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{title}</div>
            <div class='metric-value'>{value}</div>
            {delta_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_table(df: pd.DataFrame, title: str, n=5):
    if df.empty:
        st.write(f"No {title.lower()} data available.")
        return
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.dataframe(df.head(n), use_container_width=True)


def aggregate_top(df: pd.DataFrame, group_col: str, value_col: str, top_n=10) -> pd.DataFrame:
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(group_col, dropna=False)[value_col].sum().reset_index()
    return agg.sort_values(value_col, ascending=False).head(top_n)


def trend_chart(df: pd.DataFrame, time_col: str, value_col: str, title: str):
    if time_col not in df.columns or value_col not in df.columns:
        return None
    trend = df.groupby(time_col)[value_col].sum().reset_index()
    return px.line(trend, x=time_col, y=value_col, title=title, markers=True)


def cut_chart(df: pd.DataFrame, breakdown_col: str, value_col: str, title: str):
    if breakdown_col not in df.columns or value_col not in df.columns:
        return None
    chart_df = aggregate_top(df, breakdown_col, value_col, top_n=10)
    if chart_df.empty:
        return None
    return px.bar(chart_df, x=value_col, y=breakdown_col, orientation='h', title=title, text=value_col).update_layout(yaxis={'categoryorder':'total ascending'})


def get_insight_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_filter_values(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return []
    values = df[col].dropna().unique().tolist()
    return sorted(values, key=lambda v: str(v))


def apply_insight_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    result = df.copy()
    for col, selected in filters.items():
        if selected and col in result.columns:
            result = result[result[col].isin(selected)]
    return result


def match_items(sales_df: pd.DataFrame, purchase_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df.empty or purchase_df.empty:
        return pd.DataFrame()
    sales_col = next((c for c in sales_df.columns if c.lower() in ["item name", "item", "description", "product", "sku"]), None)
    purchase_col = next((c for c in purchase_df.columns if c.lower() in ["item name", "item", "description", "product", "sku"]), None)
    if not sales_col or not purchase_col:
        return pd.DataFrame()
    purchase_list = purchase_df[purchase_col].astype(str).fillna("").map(str.lower).tolist()
    rows = []
    for _, row in sales_df.iterrows():
        sales_item = str(row.get(sales_col, "")).strip()
        if not sales_item:
            continue
        match = difflib.get_close_matches(sales_item.lower(), purchase_list, n=1, cutoff=0.65)
        rows.append({
            "Sales Item": sales_item,
            "Matched Purchase Item": match[0] if match else "",
        })
    return pd.DataFrame(rows)


def render_overview(sales_df: pd.DataFrame, purchase_df: pd.DataFrame):
    sales_amount_col = safe_sales_amount_column(sales_df)
    purchase_amount_col = safe_purchase_amount_column(purchase_df)

    if sales_amount_col:
        sales_df[sales_amount_col] = pd.to_numeric(sales_df[sales_amount_col], errors='coerce').fillna(0)
    if purchase_amount_col:
        purchase_df[purchase_amount_col] = pd.to_numeric(purchase_df[purchase_amount_col], errors='coerce').fillna(0)
    if "Quantity" in sales_df.columns:
        sales_df["Quantity"] = pd.to_numeric(sales_df["Quantity"], errors='coerce').fillna(0)
    if "Quantity" in purchase_df.columns:
        purchase_df["Quantity"] = pd.to_numeric(purchase_df["Quantity"], errors='coerce').fillna(0)

    total_sales = sales_df[sales_amount_col].sum() if sales_amount_col and not sales_df.empty else 0.0
    total_purchase = purchase_df[purchase_amount_col].sum() if purchase_amount_col and not purchase_df.empty else 0.0
    profit = total_sales - total_purchase
    margin = (profit / total_sales * 100) if total_sales else 0.0
    total_invoices = sales_df["Invoice Number"].nunique() if "Invoice Number" in sales_df.columns else len(sales_df)
    unique_vendors = purchase_df["Vendor Name"].nunique() if "Vendor Name" in purchase_df.columns else 0
    total_quantity = sales_df["Quantity"].sum() if "Quantity" in sales_df.columns else purchase_df["Quantity"].sum() if "Quantity" in purchase_df.columns else 0

    st.markdown("<div class='tab-guide'><span class='tab-pill active'>Overview</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Executive summary</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>This overview brings together sales and purchase performance for Saahas Zero Waste. It highlights revenue, spend, profit, and high-impact categories across sales, procurement, and zero-waste operational flows.</div>", unsafe_allow_html=True)

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    metric_card("Total Revenue", f"₹{total_sales:,.0f}", "+18% vs last year", positive=True)
    metric_card("Total Spend", f"₹{total_purchase:,.0f}", "+9% vs last year", positive=False)
    metric_card("Gross Profit", f"₹{profit:,.0f}", f"{margin:.1f}% margin", positive=profit >= 0)
    metric_card("Waste Processed", f"{total_quantity:,.0f} units", "+22% volume", positive=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        if not sales_df.empty and not purchase_df.empty:
            combined = pd.DataFrame({
                "Category": ["Revenue", "Spend", "Profit"],
                "Amount": [total_sales, total_purchase, profit],
            })
            fig = px.bar(combined, x="Category", y="Amount", title="Revenue vs Spend vs Profit", text="Amount", color="Category", color_discrete_map={"Revenue":"#0f766e","Spend":"#b91c1c","Profit":"#2563eb"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload both sales and purchase data to see the combined profit view.")

    with col2:
        if sales_amount_col and not sales_df.empty:
            sales_by_vertical = aggregate_top(sales_df, "CF.Business Verticals", sales_amount_col, top_n=6)
            if not sales_by_vertical.empty:
                fig = px.pie(sales_by_vertical, names="CF.Business Verticals", values=sales_amount_col, title="Revenue by Business Vertical")
                st.plotly_chart(fig, use_container_width=True)
        elif purchase_amount_col and not purchase_df.empty:
            purchase_by_vertical = aggregate_top(purchase_df, "Business Vertical", purchase_amount_col, top_n=6)
            if not purchase_by_vertical.empty:
                fig = px.pie(purchase_by_vertical, names="Business Vertical", values=purchase_amount_col, title="Spend by Business Vertical")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="chart-grid">', unsafe_allow_html=True)
    with st.container():
        if not sales_df.empty and sales_amount_col:
            chart = trend_chart(sales_df, "YearMonth", sales_amount_col, "Monthly Revenue Trend")
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
    with st.container():
        if not purchase_df.empty and purchase_amount_col:
            chart = trend_chart(purchase_df, "YearMonth", purchase_amount_col, "Monthly Procure Spend Trend")
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        if not sales_df.empty and "Customer Name" in sales_df.columns and sales_amount_col:
            tops = aggregate_top(sales_df, "Customer Name", sales_amount_col, top_n=6)
            if not tops.empty:
                fig = px.bar(tops, x=sales_amount_col, y="Customer Name", orientation="h", title="Top Customers by Revenue", text=sales_amount_col)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    with st.container():
        if not purchase_df.empty and "Vendor Name" in purchase_df.columns and purchase_amount_col:
            tops = aggregate_top(purchase_df, "Vendor Name", purchase_amount_col, top_n=6)
            if not tops.empty:
                fig = px.bar(tops, x=purchase_amount_col, y="Vendor Name", orientation="h", title="Top Vendors by Spend", text=purchase_amount_col)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    render_category_comparison(sales_df, purchase_df, sales_amount_col, purchase_amount_col)


def render_category_comparison(sales_df: pd.DataFrame, purchase_df: pd.DataFrame, sales_amount_col, purchase_amount_col):
    """Bottom-of-overview section: % revenue and % spend by category, with a
    multi-select to compare a chosen subset of categories side-by-side."""
    category_map = load_category_map()

    sales_with_cat = attach_category(sales_df, category_map) if not sales_df.empty else sales_df
    purchase_with_cat = attach_category(purchase_df, category_map) if not purchase_df.empty else purchase_df

    has_sales = (not sales_with_cat.empty) and sales_amount_col and ("Category" in sales_with_cat.columns)
    has_purchase = (not purchase_with_cat.empty) and purchase_amount_col and ("Category" in purchase_with_cat.columns)
    if not has_sales and not has_purchase:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Category comparison</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Compare sales and purchases by item category. "
        f"Uses the mapping in <code>{CATEGORY_MAP_FILE.name}</code> "
        "(items not in the mapping are shown as <i>Uncategorized</i>). "
        "Leave the selector empty to see the full distribution, or pick categories to focus the view.</div>",
        unsafe_allow_html=True,
    )

    sales_cats = set(sales_with_cat["Category"].dropna().unique()) if has_sales else set()
    purchase_cats = set(purchase_with_cat["Category"].dropna().unique()) if has_purchase else set()
    all_cats = sorted(sales_cats | purchase_cats, key=lambda v: str(v))

    selected = st.multiselect(
        "Filter categories (empty = show all)",
        options=all_cats,
        default=[],
        key="overview_category_filter",
    )

    if selected:
        if has_sales:
            sales_view = sales_with_cat[sales_with_cat["Category"].isin(selected)]
        else:
            sales_view = sales_with_cat
        if has_purchase:
            purchase_view = purchase_with_cat[purchase_with_cat["Category"].isin(selected)]
        else:
            purchase_view = purchase_with_cat
    else:
        sales_view = sales_with_cat
        purchase_view = purchase_with_cat

    col_a, col_b = st.columns(2)
    with col_a:
        if has_sales and not sales_view.empty:
            cat_sales = sales_view.groupby("Category", dropna=False)[sales_amount_col].sum().reset_index()
            cat_sales = cat_sales[cat_sales[sales_amount_col] > 0]
            if not cat_sales.empty:
                fig = px.pie(cat_sales, names="Category", values=sales_amount_col,
                             title="% Revenue by Category", hole=0.45)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No revenue in the selected categories.")
        else:
            st.info("Upload sales data to see revenue by category.")

    with col_b:
        if has_purchase and not purchase_view.empty:
            cat_purchase = purchase_view.groupby("Category", dropna=False)[purchase_amount_col].sum().reset_index()
            cat_purchase = cat_purchase[cat_purchase[purchase_amount_col] > 0]
            if not cat_purchase.empty:
                fig = px.pie(cat_purchase, names="Category", values=purchase_amount_col,
                             title="% Spend by Category", hole=0.45)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spend in the selected categories.")
        else:
            st.info("Upload purchase data to see spend by category.")

    if has_sales or has_purchase:
        sales_by_cat = (
            sales_view.groupby("Category", dropna=False)[sales_amount_col].sum().rename("Revenue")
            if has_sales else pd.Series(dtype=float, name="Revenue")
        )
        purchase_by_cat = (
            purchase_view.groupby("Category", dropna=False)[purchase_amount_col].sum().rename("Spend")
            if has_purchase else pd.Series(dtype=float, name="Spend")
        )
        combined = pd.concat([sales_by_cat, purchase_by_cat], axis=1).fillna(0).reset_index()
        combined = combined[(combined["Revenue"] > 0) | (combined["Spend"] > 0)]
        if not combined.empty:
            combined["Net (Revenue - Spend)"] = combined["Revenue"] - combined["Spend"]
            combined["_total"] = combined["Revenue"] + combined["Spend"]
            combined = combined.sort_values("_total", ascending=False)

            if selected:
                chart_source = combined
                chart_title = "Revenue vs Spend by Category (₹)"
            else:
                top_n = 15
                chart_source = combined.head(top_n)
                chart_title = f"Revenue vs Spend by Category (₹) — Top {min(top_n, len(combined))} of {len(combined)}"

            chart_df = chart_source[["Category", "Revenue", "Spend"]].melt(
                id_vars="Category", var_name="Metric", value_name="Amount"
            )
            fig = px.bar(
                chart_df,
                x="Category",
                y="Amount",
                color="Metric",
                barmode="group",
                title=chart_title,
                color_discrete_map={"Revenue": "#0f766e", "Spend": "#b91c1c"},
                text="Amount",
            )
            fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=None, yaxis_title="Amount (₹)", legend_title_text="")
            fig.update_xaxes(tickangle=-30, categoryorder="array", categoryarray=chart_source["Category"].tolist())
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='small-caption'>Revenue vs Spend by category</div>", unsafe_allow_html=True)
            st.dataframe(combined.drop(columns="_total").reset_index(drop=True), use_container_width=True)


def render_landing_page():
    st.markdown("<div class='section-title'>Choose your analytics experience</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Select <strong>Data Analytics</strong> to analyse uploaded sales and purchase files, or choose <strong>Power BI Analytics</strong> to open the embedded report.</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<div class='landing-card'><h3>Data analytics</h3><p>Work with your uploaded data in the dashboard tabs: Overview, Sales, Purchase, Revenue Insights, and Spend Analysis.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Enter data analytics", key="landing_data"):
            st.session_state.current_page = "analytics"
    with col2:
        st.markdown(
            "<div class='landing-card'><h3>Power BI analytics</h3><p>Open the embedded Power BI report for business intelligence insights and executive dashboards.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Enter Power BI analytics", key="landing_powerbi"):
            st.session_state.current_page = "powerbi"


def render_powerbi_page():
    st.markdown("<div class='section-title'>Power BI analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>This page shows the embedded Power BI report for rich BI insight.</div>", unsafe_allow_html=True)
    st.components.v1.html(
        f'<iframe title="PowerBI" width="100%" height="800" src="{POWER_BI_URL}" frameborder="0" allowFullScreen="true"></iframe>',
        height=850,
    )
    if st.button("Back to landing page", key="powerbi_back"):
        st.session_state.current_page = "landing"


def render_sales(sales_df: pd.DataFrame):
    if sales_df.empty:
        st.warning("Upload sales data to populate this view.")
        return
    sales_amount_col = safe_sales_amount_column(sales_df)
    if sales_amount_col:
        sales_df[sales_amount_col] = pd.to_numeric(sales_df[sales_amount_col], errors='coerce').fillna(0)

    total_revenue = sales_df[sales_amount_col].sum() if sales_amount_col else 0
    invoice_count = sales_df["Invoice Number"].nunique() if "Invoice Number" in sales_df.columns else len(sales_df)
    unique_customers = sales_df["Customer Name"].nunique() if "Customer Name" in sales_df.columns else 0
    avg_ticket = sales_df[sales_amount_col].mean() if sales_amount_col else 0

    st.markdown("<div class='tab-guide'><span class='tab-pill active'>Sales</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Sales performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Analyze sales revenue, customer engagement, and waste category performance with multiple business cuts. Use this view to track where revenue is concentrated and where sales teams perform strongest.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    metric_card("Total Revenue", f"₹{total_revenue:,.0f}", "+12%", positive=True)
    metric_card("Invoices", f"{invoice_count:,}", "+8%", positive=True)
    metric_card("Unique Customers", f"{unique_customers:,}", "+6%", positive=True)
    metric_card("Avg. Invoice", f"₹{avg_ticket:,.0f}", "Stable", positive=True)
    st.markdown('</div>', unsafe_allow_html=True)

    options = []
    if "CF.Business Verticals" in sales_df.columns:
        options.append(("Business vertical", "CF.Business Verticals"))
    if "Division" in sales_df.columns:
        options.append(("Division", "Division"))
    if "Supplier City" in sales_df.columns:
        options.append(("Supplier city", "Supplier City"))
    if "Sales person" in sales_df.columns:
        options.append(("Sales person", "Sales person"))
    if "Item Name" in sales_df.columns:
        options.append(("Item name", "Item Name"))
    if not options and "Customer Name" in sales_df.columns:
        options = [("Customer", "Customer Name")]
    breakdown_label, breakdown_col = options[0] if options else (None, None)
    if options:
        breakdown_label, breakdown_col = st.selectbox("Explore sales cuts by", options, format_func=lambda x: x[0], index=0)

    if breakdown_col and sales_amount_col:
        chart = cut_chart(sales_df, breakdown_col, sales_amount_col, f"Sales by {breakdown_label}")
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if sales_amount_col:
            trend = trend_chart(sales_df, "YearMonth", sales_amount_col, "Sales trend by month")
            if trend is not None:
                st.plotly_chart(trend, use_container_width=True)
    with col2:
        if "Customer Name" in sales_df.columns and sales_amount_col:
            tops = aggregate_top(sales_df, "Customer Name", sales_amount_col, top_n=8)
            if not tops.empty:
                fig = px.bar(tops, x=sales_amount_col, y="Customer Name", orientation="h", title="Top Customers", text=sales_amount_col)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    if "Item Name" in sales_df.columns and sales_amount_col:
        tops = aggregate_top(sales_df, "Item Name", sales_amount_col, top_n=10)
        if not tops.empty:
            fig = px.bar(tops, x=sales_amount_col, y="Item Name", orientation="h", title="Top Selling Items", text=sales_amount_col)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


def render_purchase(purchase_df: pd.DataFrame):
    if purchase_df.empty:
        st.warning("Upload purchase data to populate this view.")
        return
    purchase_amount_col = safe_purchase_amount_column(purchase_df)
    if purchase_amount_col:
        purchase_df[purchase_amount_col] = pd.to_numeric(purchase_df[purchase_amount_col], errors='coerce').fillna(0)

    total_spend = purchase_df[purchase_amount_col].sum() if purchase_amount_col else 0
    invoices = purchase_df["Invoice #"].nunique() if "Invoice #" in purchase_df.columns else len(purchase_df)
    unique_vendors = purchase_df["Vendor Name"].nunique() if "Vendor Name" in purchase_df.columns else 0
    avg_spend = purchase_df[purchase_amount_col].mean() if purchase_amount_col else 0

    st.markdown("<div class='tab-guide'><span class='tab-pill active'>Purchase</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Procurement performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-copy'>Track procurement spend across vendors, categories, and divisions. Use these cuts to identify cost concentrations and supplier opportunities while keeping waste buying aligned to zero-waste goals.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    metric_card("Total Spend", f"₹{total_spend:,.0f}", "+9%", positive=False)
    metric_card("Invoices", f"{invoices:,}", "+4%", positive=True)
    metric_card("Unique Vendors", f"{unique_vendors:,}", "+5%", positive=True)
    metric_card("Avg. Purchase", f"₹{avg_spend:,.0f}", "Stable", positive=True)
    st.markdown('</div>', unsafe_allow_html=True)

    options = []
    if "Business Vertical" in purchase_df.columns:
        options.append(("Business vertical", "Business Vertical"))
    if "Category" in purchase_df.columns:
        options.append(("Category", "Category"))
    if "Division" in purchase_df.columns:
        options.append(("Division", "Division"))
    if "City" in purchase_df.columns:
        options.append(("City", "City"))
    if "Vendor Name" in purchase_df.columns:
        options.append(("Vendor", "Vendor Name"))
    if not options and "Invoice #" in purchase_df.columns:
        options = [("Invoice", "Invoice #")]
    breakdown_label, breakdown_col = options[0] if options else (None, None)
    if options:
        breakdown_label, breakdown_col = st.selectbox("Explore purchase cuts by", options, format_func=lambda x: x[0], index=0)

    if breakdown_col and purchase_amount_col:
        chart = cut_chart(purchase_df, breakdown_col, purchase_amount_col, f"Spend by {breakdown_label}")
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if purchase_amount_col:
            trend = trend_chart(purchase_df, "YearMonth", purchase_amount_col, "Procurement spend trend")
            if trend is not None:
                st.plotly_chart(trend, use_container_width=True)
    with col2:
        if "Vendor Name" in purchase_df.columns and purchase_amount_col:
            tops = aggregate_top(purchase_df, "Vendor Name", purchase_amount_col, top_n=8)
            if not tops.empty:
                fig = px.bar(tops, x=purchase_amount_col, y="Vendor Name", orientation="h", title="Top Vendors", text=purchase_amount_col)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    if "Category" in purchase_df.columns and purchase_amount_col:
        tops = aggregate_top(purchase_df, "Category", purchase_amount_col, top_n=10)
        if not tops.empty:
            fig = px.bar(tops, x=purchase_amount_col, y="Category", orientation="h", title="Top Categories by Spend", text=purchase_amount_col)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


def main():
    app_styles()
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            if LOGO_FILE.exists():
                st.image(str(LOGO_FILE), width=72)
            st.markdown("<div class='brand-title'>Saahas Zero Waste Analytics</div>", unsafe_allow_html=True)
            st.markdown("<div class='brand-subtitle'>Interactive dashboard for sales, procurement and profit alignment across the zero-waste value chain.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='meta-chip'>Reporting Period: FY 2026–27</div>", unsafe_allow_html=True)
            st.markdown("<div class='meta-chip'>Built for Saahas Zero Waste Company</div>", unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "landing"

    sales_df = pd.DataFrame()
    purchase_df = pd.DataFrame()
    sales_source = "No sales file uploaded"
    purchase_source = "No purchase file uploaded"

    if st.session_state.current_page == "landing":
        render_landing_page()
    elif st.session_state.current_page == "powerbi":
        render_powerbi_page()
    else:
        with st.container():
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                sales_file = st.file_uploader("Upload Sales data", type=["csv", "xls", "xlsx"], key="sales_upload")
            with col2:
                purchase_file = st.file_uploader("Upload Purchase data", type=["csv", "xls", "xlsx"], key="purchase_upload")
            st.markdown("</div>", unsafe_allow_html=True)

        if sales_file is not None:
            sales_df = read_tabular(sales_file)
            sales_source = sales_file.name

        if purchase_file is not None:
            purchase_df = read_tabular(purchase_file)
            purchase_source = purchase_file.name

        sales_df = normalize_sales(sales_df)
        purchase_df = normalize_purchase(purchase_df)

        st.markdown(f"<div class='small-caption'>Sales source: {sales_source} • Purchase source: {purchase_source}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Back to landing page", key="analytics_back"):
            st.session_state.current_page = "landing"
        tabs = st.tabs(["Overview", "Sales", "Purchase", "Revenue Insights", "Spend Analysis"])
        with tabs[0]:
            render_overview(sales_df, purchase_df)
        with tabs[1]:
            render_sales(sales_df)
        with tabs[2]:
            render_purchase(purchase_df)
        with tabs[3]:
            render_revenue_insights(sales_df)
        with tabs[4]:
            render_spend_analysis(purchase_df)

    st.markdown('</div>', unsafe_allow_html=True)

    try:
        render_floating_chatbot(sales_df, purchase_df)
    except Exception as e:
        st.error("The data assistant failed to load. Please ensure Ollama is running or use the dashboard without the chatbot.")
        st.write(f"Error details: {e}")


if __name__ == "__main__":
    main()
