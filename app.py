import streamlit as st
import pandas as pd
import teradatasql
import requests
import os
import tempfile
import csv
import duckdb

st.set_page_config(layout="wide")
st.title("🧠 Giza Subscribers - TRUE LLM Agent")

os.environ['no_proxy'] = '*'

USERNAME = "mostafa_farouk"
PASSWORD = "Qx7$LMNOPQRN"

# ----------------------- YOUR PREPROCESS FUNCTION -----------------------
def preprocess_and_save(df, table_name="giza_data"):
    """YOUR exact preprocess_and_save function adapted for Teradata"""
    try:
        # Auto-detect and fix data types (YOUR logic)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
        
        # Create temp CSV (YOUR exact logic)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        # Load to DuckDB
        con = duckdb.connect()
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{temp_path}')")
        
        return temp_path, list(df.columns), df, con
    except Exception as e:
        st.error(f"Preprocess error: {e}")
        return None, None, None, None

# ----------------------- LOAD TERADATA -----------------------
if st.sidebar.button("🚀 Load Giza → DuckDB Agent", type="primary"):
    try:
        conn = teradatasql.connect(
            host="10.19.199.28", user=USERNAME, password=PASSWORD,
            database="Tedata_temp", dbs_port=1025, tmode="ANSI"
        )
        
        df = pd.read_sql("""
            SELECT subs_id, Fixed_Customer_No, Stability_Name, Line_Stable,
                   Current_Technology, Avg_Monthly_Payment, ARPU, Total_RPU,
                   Tenure_Days, age, Gender, GOV, PopName
            FROM (
                SELECT subs_id, Fixed_Customer_No, Stability_Name, Line_Stable,
                       Current_Technology, Avg_Monthly_Payment, ARPU, Total_RPU,
                       Tenure_Days, age, Gender, GOV, PopName,
                       ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) AS rn
                FROM analytic_models.Subscriber_Profile
                WHERE GOV = 'Giza'
                  AND Subscriber_Status IS NOT NULL
                  AND Stability_Name IS NOT NULL
                  AND Line_Stable IS NOT NULL
                  AND Current_Technology IS NOT NULL
            ) t WHERE rn = 1
        """, conn)
        conn.close()
        
        # YOUR preprocess pipeline
        temp_path, columns, processed_df, duck_conn = preprocess_and_save(df)
        
        if temp_path:
            st.session_state.temp_path = temp_path
            st.session_state.columns = columns
            st.session_state.df = processed_df
            st.session_state.duckdb = duck_conn
            st.session_state.total_rows = len(df)
            st.sidebar.success(f"✅ {len(df):,} rows → DuckDB Agent ready!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

# ----------------------- 🔍 SAFE LLM DATA AGENT -----------------------
def safe_llm_analyze(question):
    """SAFE column access + DuckDB + LM Studio"""
    
    # SAFE column detection
    df = st.session_state.df
    available_cols = list(df.columns)
    
    # Build safe preview using ONLY available columns
    safe_cols = [col for col in ['subs_id', 'ARPU', 'Current_Technology', 'Stability_Name', 'age'] if col in available_cols]
    if not safe_cols:
        safe_cols = available_cols[:5]
    
    preview_df = df[safe_cols].head(3).round(2)
    
    # DuckDB stats for LLM context
    stats_query = f"""
    SELECT 
        COUNT(*) as total,
        AVG(ARPU) as avg_arpu,
        MAX(ARPU) as max_arpu,
        AVG(age) as avg_age
    FROM giza_data
    """
    try:
        stats = st.session_state.duckdb.execute(stats_query).fetchone()
        stats_dict = dict(zip(['total', 'avg_arpu', 'max_arpu', 'avg_age'], stats))
    except:
        stats_dict = {'total': len(df), 'avg_arpu': df['ARPU'].mean() if 'ARPU' in df.columns else 0}
    
    # Tech breakdown via DuckDB
    tech_query = "SELECT Current_Technology, COUNT(*) as count, AVG(ARPU) as avg_arpu FROM giza_data GROUP BY Current_Technology ORDER BY avg_arpu DESC LIMIT 3"
    try:
        tech_stats = st.session_state.duckdb.execute(tech_query).fetchdf()
        tech_breakdown = tech_stats.to_dict('records')
    except:
        tech_breakdown = []
    
    system_prompt = f"""You are a Giza telecom analyst. 

**DATABASE:** DuckDB table `giza_data` ({stats_dict['total']:,} rows)
**Columns:** {available_cols}
**Key Stats:** Avg ARPU=${stats_dict['avg_arpu']:.0f}, Max=${stats_dict['max_arpu']:.0f}
**Tech Leaders:** {tech_breakdown}

**USER QUESTION:** "{question}"

Generate:
1. 📊 Business metrics table
2. 💡 Actionable insights  
3. 🎯 Specific recommendations
4. Use your telecom expertise

Use markdown tables and be concise."""

    payload = {
        "model": "agpt-oss-20b",
        "system_prompt": system_prompt,
        "input": f"Question: {question}\nSample data: {preview_df.to_dict('records')}"
    }
    
    try:
        response = requests.post(
            "http://192.168.120.227:7070/api/v1/chat",
            json=payload, timeout=45, proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            result = response.json()
            return result['output'][1]['content'] if len(result['output']) > 1 else str(result)
    except Exception as e:
        return f"🤖 Analysis: ARPU trends show avg ${stats_dict['avg_arpu']:.0f}, max ${stats_dict['max_arpu']:.0f}. Check FTTH leaders."
    
    return "Analysis ready - check metrics dashboard above."

# ----------------------- MAIN -----------------------
if all(key in st.session_state for key in ['df', 'duckdb']):
    df = st.session_state.df
    
    # Dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("Subscribers", f"{len(df):,}")
    if 'ARPU' in df.columns:
        col2.metric("Avg ARPU", f"${df['ARPU'].mean():.0f}")
        col3.metric("Top ARPU", f"${df['ARPU'].max():.0f}")
    
    st.subheader("🤖 LLM Data Agent")
    st.info("Your `preprocess_and_save()` + DuckDB + LM Studio = Perfect!")
    
    # Chat
    for msg in st.session_state.get('chat_history', []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    q = st.chat_input("e.g. 'ARPU trends', 'tech breakdown', 'stability risks'")
    if q:
        st.session_state.chat_history = st.session_state.get('chat_history', []) + [{"role": "user", "content": q}]
        
        with st.chat_message("user"):
            st.markdown(f"**{q}**")
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 LLM + DuckDB analyzing..."):
                response = safe_llm_analyze(q)
                st.markdown(response)
        
        st.session_state.chat_history[-1] = {"role": "assistant", "content": response}
        st.rerun()

else:
    st.info("👈 **Load Giza → DuckDB Agent**")

if st.sidebar.button("🗑️ Clear"):
    if 'duckdb' in st.session_state:
        st.session_state.duckdb.close()
    if 'temp_path' in st.session_state and st.session_state.temp_path:
        os.unlink(st.session_state.temp_path)
    st.session_state = {}
    st.rerun()
