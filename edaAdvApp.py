import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import textwrap
import re
from typing import Optional

st.set_page_config(page_title="My EDA App", layout="wide")

# --- Helpers -------------------------------------------------
@st.cache_data
def load_excel(file) -> dict:
    try:
        xls = pd.ExcelFile(file)
        sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
        return sheets
    except Exception:
        return {}

@st.cache_data
def read_csv(file, **kwargs):
    return pd.read_csv(file, **kwargs)


def df_info_to_string(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def find_column_by_keywords(df: pd.DataFrame, keywords: list) -> Optional[str]:
    lowcols = {c.lower(): c for c in df.columns}
    for k in keywords:
        for lc, orig in lowcols.items():
            if k in lc:
                return orig
    return None


def parse_user_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return pd.DataFrame()

    # top N categories (e.g. "show me top 5 categories" or "top 3 product_category")
    m = re.search(r'top\s*(\d+)\s*(?:categories|category|distinct|items|values)?(?:\s*(?:in|of)\s*([a-zA-Z0-9 _]+))?', q)
    if m:
        n = int(m.group(1))
        col = m.group(2)
        if col:
            col = col.strip()
            # try to match col to actual column name
            match = None
            for c in df.columns:
                if c.lower() == col.lower() or col.lower() in c.lower():
                    match = c
                    break
            if not match:
                # fallback: first categorical-like column
                cat = df.select_dtypes(include=['object','category']).columns
                match = cat[0] if len(cat)>0 else df.columns[0]
        else:
            # choose first categorical-like column
            cat = df.select_dtypes(include=['object','category']).columns
            match = cat[0] if len(cat)>0 else df.columns[0]
        result = df[match].value_counts().head(n).rename_axis(match).reset_index(name='count')
        return result

    # conditional: show me those records where ... more than 5 customer service calls
    # look for pattern "more than X" and keywords like customer, service, call(s)
    m2 = re.search(r'(?:more than|greater than|>)\s*(\d+)', q)
    if m2 and ('customer' in q or 'service' in q or 'call' in q):
        val = float(m2.group(1))
        # find best candidate column
        col = find_column_by_keywords(df, ['customer','service','call','calls'])
        if not col:
            # fallback to any numeric column
            numeric = df.select_dtypes(include=[np.number]).columns
            if len(numeric)>0:
                col = numeric[0]
            else:
                return pd.DataFrame()
        try:
            return df[pd.to_numeric(df[col], errors='coerce') > val]
        except Exception:
            return pd.DataFrame()

    # generic "where <col> > <value>" pattern
    m3 = re.search(r'where\s+([a-zA-Z0-9 _]+)\s*(>=|<=|>|<|=|==)\s*([0-9\.]+)', q)
    if m3:
        rawcol = m3.group(1).strip()
        op = m3.group(2)
        val = float(m3.group(3))
        match = None
        for c in df.columns:
            if rawcol.lower() == c.lower() or rawcol.lower() in c.lower():
                match = c
                break
        if match:
            ser = pd.to_numeric(df[match], errors='coerce')
            if op in ('>', 'greater than'):
                return df[ser > val]
            if op == '<':
                return df[ser < val]
            if op in ('=', '=='):
                return df[ser == val]
            if op == '>=':
                return df[ser >= val]
            if op == '<=':
                return df[ser <= val]
    # fallback: return empty
    return pd.DataFrame()

# --- UI ------------------------------------------------------
st.title("My EDA App")
st.markdown("Upload a CSV or Excel file, explore, visualize, and run simple queries.")

with st.sidebar.expander("Upload & options", expanded=True):
    file = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx','xls'])
    show_full_info = st.checkbox("Show full df.info() output", value=False)
    sample_rows = st.number_input("Preview rows", min_value=3, max_value=100, value=5)
    auto_select_cols = st.checkbox("Auto-select all columns by default", value=True)

if file is not None:
    df = None
    if file.name.lower().endswith(('.xls','.xlsx')):
        sheets = load_excel(file)
        sheet_names = list(sheets.keys())
        sheet = st.sidebar.selectbox("Select sheet", sheet_names)
        df = sheets[sheet]
    else:
        df = read_csv(file)

    # show head & info & describe
    st.subheader("Preview & Overview")
    left, right = st.columns((2,1))
    with left:
        st.dataframe(df.head(sample_rows))
        st.write(f"Rows: {df.shape[0]}  —  Columns: {df.shape[1]}")
    with right:
        with st.expander("DataFrame info"):
            if show_full_info:
                st.text(df_info_to_string(df))
            else:
                st.write(df.dtypes)
        with st.expander("Descriptive stats (numeric)"):
            st.dataframe(df.describe().T)
        with st.expander("Missing values & duplicates"):
            miss = df.isnull().sum().rename('missing_count')
            miss = miss[miss>0].sort_values(ascending=False)
            st.dataframe(miss)
            st.write("Duplicate rows:", int(df.duplicated().sum()))
            if st.button("Drop duplicate rows"):
                df = df.drop_duplicates()
                st.success("Dropped duplicates")

    # column multiselect
    st.subheader("Choose columns to analyze")
    cols = df.columns.tolist()
    default = cols if auto_select_cols else []
    sel_cols = st.multiselect("Select columns", options=cols, default=default)
    if len(sel_cols) == 0:
        st.info("Select at least one column to enable plotting and column-specific analysis.")

    # visualizations
    st.subheader("Visualizations")
    chart_type = st.selectbox("Chart type", [
        'Histogram (numeric)', 'Boxplot (numeric)', 'Countplot (categorical)', 'Scatter plot', 'Correlation heatmap'
    ])

    if chart_type == 'Histogram (numeric)':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col = st.selectbox('Numeric column', num_cols)
        bins = st.slider('Bins', 5, 100, 30)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), bins=bins, ax=ax)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)

    if chart_type == 'Boxplot (numeric)':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col = st.selectbox('Numeric column for boxplot', num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)

    if chart_type == 'Countplot (categorical)':
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        col = st.selectbox('Categorical column', cat_cols)
        top_n = st.slider('Top categories to show', 3, 30, 10)
        vc = df[col].value_counts().nlargest(top_n)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel('count')
        ax.set_title(f'Top {top_n} categories in {col}')
        st.pyplot(fig)

    if chart_type == 'Scatter plot':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            xcol = st.selectbox('X axis', num_cols, index=0)
            ycol = st.selectbox('Y axis', num_cols, index=1)
            hue_candidates = df.select_dtypes(include=['object','category']).columns.tolist()
            hue = st.selectbox('Hue (optional)', ['None'] + hue_candidates)
            fig, ax = plt.subplots()
            if hue != 'None':
                sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue, ax=ax)
            else:
                sns.scatterplot(data=df, x=xcol, y=ycol, ax=ax)
            ax.set_title(f'{ycol} vs {xcol}')
            st.pyplot(fig)
        else:
            st.warning('Need at least two numeric columns for scatter plot')

    if chart_type == 'Correlation heatmap':
        num_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', ax=ax)
        ax.set_title('Correlation matrix')
        st.pyplot(fig)

    # user query
    st.subheader('Run a quick user-style query')
    st.markdown('Examples: `show me top 5 categories in product_category`, `show me those records where customer initiated more than 5 customer service calls`, `where Age > 30`')
    user_query = st.text_input('Type a simple query here')
    if st.button('Run query'):
        if not user_query:
            st.warning('Please type a query')
        else:
            res = parse_user_query(user_query, df)
            if res is None or res.empty:
                st.info('No results or unable to parse query. Try: "top 5 product_category" or "where Age > 30"')
            else:
                st.write('Query result:')
                st.dataframe(res.head(100))
                csv = res.to_csv(index=False).encode('utf-8')
                st.download_button('Download result as CSV', data=csv, file_name='query_result.csv')

    st.markdown('---')
    st.caption('Advanced version: more interactive layout and controls. Uses seaborn + matplotlib. Copy logic into advanced_app.py and run with streamlit.')

else:
    st.info('Upload a CSV or Excel file from the sidebar to get started (Advanced).')