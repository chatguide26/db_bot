"""
Enhanced Streamlit App with Agno Agent + Knowledge Base + Teradata Integration
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import logging
from typing import Optional

# Local imports
from src.connectors import TeradataConnector, DuckDBConnector, SmartDataProcessor
from src.knowledge_manager import KnowledgeManager
from src.agent_tools import create_tool_functions
from src.visualizations import ChartBuilder, ChartType, VisualizationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
TERADATA_CONFIG = {
    "host": os.getenv("TERADATA_HOST", "10.19.199.28"),
    "username": os.getenv("TERADATA_USER", "mostafa_farouk"),
    "password": os.getenv("TERADATA_PASSWORD", "Qx7$LMNOPQRN"),
    "database": os.getenv("TERADATA_DB", "Tedata_temp")
}

BASE_QUERY = """
SELECT subs_id, Fixed_Customer_No, Stability_Name, Line_Stable,
       Current_Technology, Avg_Monthly_Payment, ARPU, Total_RPU,
       Tenure_Days, age, Gender, GOV, PopName,
       Subscriber_Status, Insertion_Date
FROM (
    SELECT subs_id, Fixed_Customer_No, Stability_Name, Line_Stable,
           Current_Technology, Avg_Monthly_Payment, ARPU, Total_RPU,
           Tenure_Days, age, Gender, GOV, PopName, Subscriber_Status,
           Insertion_Date,
           ROW_NUMBER() OVER (PARTITION BY subs_id ORDER BY Insertion_Date DESC) AS rn
    FROM analytic_models.Subscriber_Profile
    WHERE GOV = '{gov}'
      AND Subscriber_Status IS NOT NULL
) t WHERE rn = 1
"""

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧠 Telecom Analytics Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI-Powered Telecom Analytics Agent")
st.markdown("**Teradata → DuckDB → Agno LLM = Business Intelligence**")

# ==================== SESSION STATE ====================
if "connection" not in st.session_state:
    st.session_state.connection = None
if "duckdb" not in st.session_state:
    st.session_state.duckdb = None
if "data" not in st.session_state:
    st.session_state.data = None
if "knowledge_manager" not in st.session_state:
    st.session_state.knowledge_manager = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tools" not in st.session_state:
    st.session_state.tools = None

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Governorate selection
    gov = st.selectbox("📍 Select Governorate", ["Giza", "Cairo", "Alexandria", "Qalyubia"])
    
    # Connection button
    if st.button("🚀 Initialize Agent", type="primary"):
        with st.spinner("🔄 Connecting to Teradata..."):
            try:
                # Connect to Teradata
                teradata = TeradataConnector(**TERADATA_CONFIG)
                if teradata.connect():
                    # Query data
                    query = BASE_QUERY.format(gov=gov)
                    df = teradata.query(query)
                    teradata.close()
                    
                    if df is not None and len(df) > 0:
                        # Preprocess data
                        df = SmartDataProcessor.preprocess_subscriber_data(df)
                        
                        # Load to DuckDB
                        duckdb_conn = DuckDBConnector()
                        duckdb_conn.load_dataframe(df, "giza_data")
                        
                        # Load knowledge base
                        km = KnowledgeManager("knowledge")
                        
                        # Save to session
                        st.session_state.data = df
                        st.session_state.duckdb = duckdb_conn
                        st.session_state.knowledge_manager = km
                        st.session_state.tools = create_tool_functions(duckdb_conn, km)
                        
                        st.success(f"✅ Agent initialized with {len(df):,} {gov} subscribers!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ No data returned from Teradata")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Info section
    st.subheader("📚 Knowledge Base")
    if st.session_state.knowledge_manager:
        km = st.session_state.knowledge_manager
        st.write(f"📊 Tables: {len(km.tables)}")
        st.write(f"📋 Metrics: {len(km.business_rules)}")
        st.write(f"🔍 Patterns: {len(km.query_patterns)}")
        st.write(f"💡 Learnings: {len(km.learnings)}")
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear Session"):
        if st.session_state.duckdb:
            st.session_state.duckdb.close()
        st.session_state = {}
        st.rerun()

# ==================== MAIN CONTENT ====================
if st.session_state.data is not None:
    df = st.session_state.data
    km = st.session_state.knowledge_manager
    
    # Dashboard Overview
    st.subheader("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Subscribers", f"{len(df):,}")
    with col2:
        avg_arpu = df['ARPU'].mean() if 'ARPU' in df.columns else 0
        st.metric("Avg ARPU (EGP)", f"₵{avg_arpu:.0f}")
    with col3:
        if 'Stability_Name' in df.columns:
            stable = (df['Stability_Name'] == 'Stable').sum()
            st.metric("Stable Subscribers", f"{stable:,}")
        else:
            st.metric("Stable Subscribers", "N/A")
    with col4:
        if 'Current_Technology' in df.columns:
            ftth = (df['Current_Technology'] == 'FTTH').sum()
            st.metric("FTTH Subscribers", f"{ftth:,}")
        else:
            st.metric("FTTH Subscribers", "N/A")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Agent Chat",
        "📊 Analytics Dashboard",
        "📋 Data Explorer",
        "💡 Knowledge Base"
    ])
    
    # ===================== TAB 1: AGENT CHAT =====================
    with tab1:
        st.subheader("💬 Intelligent Data Analyst Agent")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Agent context
        agent_context = km.build_agent_context()
        
        # Chat input
        if prompt := st.chat_input("Ask about ARPU, churn, technology, demographics..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing with knowledge base..."):
                    try:
                        # Use tools to gather data
                        tools = st.session_state.tools
                        
                        # Route question to appropriate tool
                        response, charts = _route_question(prompt, tools, km, st.session_state.duckdb)
                        
                        st.markdown(response)
                        
                        # Render visualizations
                        if charts:
                            st.markdown("---")
                            st.subheader("📊 Visualizations")
                            chart_cols = st.columns(len(charts))
                            for idx, (title, fig) in enumerate(charts):
                                with chart_cols[idx % len(chart_cols)]:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Log learning
                        logger.info(f"Question: {prompt}")
                        logger.info(f"Response generated with {len(charts)} visualizations")
                        
                    except Exception as e:
                        error_response = f"❌ Analysis failed: {str(e)}"
                        st.error(error_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_response})
    
    # ===================== TAB 2: ANALYTICS DASHBOARD =====================
    with tab2:
        st.subheader("📈 Multi-dimensional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ARPU by Technology
            if 'Current_Technology' in df.columns and 'ARPU' in df.columns:
                try:
                    tech_arpu = df.groupby('Current_Technology').agg({
                        'ARPU': ['count', 'mean', 'sum']
                    }).reset_index()
                    tech_arpu.columns = ['Technology', 'Count', 'Avg ARPU', 'Total Revenue']
                    tech_arpu = tech_arpu.sort_values('Avg ARPU', ascending=False)
                    
                    fig = px.bar(
                        tech_arpu,
                        x='Technology',
                        y='Avg ARPU',
                        color='Avg ARPU',
                        title='ARPU by Technology',
                        text='Avg ARPU'
                    )
                    fig.update_traces(texttemplate='₵%{text:.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        with col2:
            # Stability Distribution
            if 'Stability_Name' in df.columns:
                try:
                    stability = df['Stability_Name'].value_counts()
                    fig = px.pie(
                        values=stability.values,
                        names=stability.index,
                        title='Subscriber Stability Distribution',
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # ARPU by Stability
            if 'Stability_Name' in df.columns and 'ARPU' in df.columns:
                try:
                    stability_arpu = df.groupby('Stability_Name')['ARPU'].mean().sort_values(ascending=False)
                    fig = px.bar(
                        x=stability_arpu.index,
                        y=stability_arpu.values,
                        title='ARPU by Subscriber Stability',
                        labels={'x': 'Stability', 'y': 'Avg ARPU (EGP)'},
                        text=stability_arpu.values
                    )
                    fig.update_traces(texttemplate='₵%{text:.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
        
        with col4:
            # Population Distribution
            if 'PopName' in df.columns:
                try:
                    pop = df['PopName'].value_counts()
                    fig = px.pie(
                        values=pop.values,
                        names=pop.index,
                        title='Population Segment Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
    
    # ===================== TAB 3: DATA EXPLORER =====================
    with tab3:
        st.subheader("🔍 Raw Data Explorer")
        
        # Column selector
        columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=['subs_id', 'ARPU', 'Current_Technology', 'Stability_Name', 'age']
        )
        
        if columns:
            st.dataframe(
                df[columns].head(100),
                use_container_width=True,
                height=400
            )
        
        # Download option
        st.download_button(
            label="📥 Download Data (CSV)",
            data=df.to_csv(index=False),
            file_name=f"giza_subscribers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        )
    
    # ===================== TAB 4: KNOWLEDGE BASE =====================
    with tab4:
        st.subheader("📚 Knowledge Base & Learnings")
        
        kb_col1, kb_col2 = st.columns(2)
        
        with kb_col1:
            st.subheader("📊 Table Schemas")
            for table_name in km.tables.keys():
                with st.expander(f"📋 {table_name}"):
                    schema = km.tables[table_name]
                    st.write(schema.get('description', 'N/A'))
                    st.json({k: v for k, v in schema.get('columns', {}).items()})
        
        with kb_col2:
            st.subheader("🎯 Business Metrics")
            metrics_dict = km.business_rules.get('telecom_metrics', {}).get('metrics', {})
            for metric_name, metric_def in metrics_dict.items():
                with st.expander(f"📈 {metric_name}"):
                    st.write(f"**Definition:** {metric_def.get('business_value', 'N/A')}")
                    st.write(f"**Unit:** {metric_def.get('unit', 'N/A')}")
        
        st.subheader("💡 Discovered Insights")
        insights = km.get_business_insights()
        for insight in insights[:5]:
            with st.expander(f"🔍 {insight.get('insight', 'N/A')}"):
                st.write(f"**Confidence:** {insight.get('confidence', 'N/A')}")
                st.write(f"**Recommendation:** {insight.get('recommendation', 'N/A')}")

else:
    st.info("👈 **Click 'Initialize Agent' to load Giza subscriber data from Teradata**")
    st.markdown("""
    ## 🎯 What This Agent Does:
    
    1. **Connects to Teradata** - Fetches latest Giza subscriber data
    2. **Loads to DuckDB** - Ultra-fast in-memory analytics
    3. **Leverages Knowledge Base**:
       - 📊 Table schemas & relationships
       - 📋 Business metrics & definitions
       - 🔍 Query patterns that work
       - 💡 Discovered insights
    4. **Answers Natural Language Questions**:
       - "What are ARPU trends?"
       - "Which technology segment is most profitable?"
       - "Who are the at-risk high-value customers?"
    
    ## 🚀 Key Features:
    - **Layered Context**: Integrates tables, business rules, queries, learnings
    - **Intelligent Tools**: ARPU analysis, churn risk, FTTH migration, demographics
    - **Data Quality**: Auto-cleaning, deduplication, standardization
    - **Knowledge Learning**: Captures patterns and builds institutional memory
    """)

# ==================== HELPER FUNCTIONS ====================

def _route_question(prompt: str, tools: dict, km: KnowledgeManager, duckdb) -> tuple:
    """Route question to appropriate analysis tool and generate visualizations"""
    
    prompt_lower = prompt.lower()
    viz_manager = VisualizationManager()
    charts = []
    
    # Determine which tools to use
    if any(word in prompt_lower for word in ['arpu', 'revenue', 'profit']):
        arpu_results = json.loads(tools['arpu_by_technology']())
        tech_results = json.loads(tools['arpu_by_stability']())
        geo_results = json.loads(tools['arpu_by_geography']())
        
        # Generate visualizations
        arpu_df = pd.DataFrame(arpu_results['data']) if arpu_results['data'] else None
        if arpu_df is not None and not arpu_df.empty:
            fig = ChartBuilder.build(arpu_df, ChartType.BAR, 
                                    title="ARPU by Technology",
                                    x_col='Current_Technology', 
                                    y_col='avg_arpu')
            if fig:
                charts.append(("ARPU by Technology", fig))
        
        response = f"""
## 💰 ARPU Analysis

### By Technology:
{_format_table(arpu_results['data'])}

### By Stability:
{_format_table(tech_results['data'])}

### By Geography:
{_format_table(geo_results['data'])}

**Key Insight:** {arpu_results.get('insight', '')}
"""
    
    elif any(word in prompt_lower for word in ['churn', 'risk', 'at-risk']):
        risk_results = json.loads(tools['churn_risk_distribution']())
        at_risk = json.loads(tools['at_risk_segments']())
        
        # Generate visualizations
        risk_df = pd.DataFrame(risk_results['data']) if risk_results['data'] else None
        if risk_df is not None and not risk_df.empty:
            fig = ChartBuilder.build(risk_df, ChartType.PIE,
                                    title="Risk Distribution",
                                    x_col='Stability_Name',
                                    y_col='count')
            if fig:
                charts.append(("Risk Distribution", fig))
        
        response = f"""
## ⚠️ Churn Risk Analysis

### Risk Distribution:
{_format_table(risk_results['data'])}

### At-Risk Revenue:
{_format_table(at_risk['data'])}

**Recommendation:** {at_risk.get('recommendation', '')}
"""
    
    elif any(word in prompt_lower for word in ['ftth', 'migration', 'upgrade']):
        migration = json.loads(tools['ftth_migration_analysis']())
        
        # Create a gauge/indicator for migration opportunity
        migration_data = {
            'Metric': ['Migration Candidates', 'Current Avg ARPU', 'Potential Avg ARPU', 'Revenue Uplift'],
            'Value': [
                migration['data'].get('migration_candidates', 0),
                migration['data'].get('current_avg_arpu', 0),
                migration['data'].get('potential_avg_arpu', 0),
                migration['data'].get('revenue_uplift', 0)
            ]
        }
        mig_df = pd.DataFrame(migration_data)
        
        response = f"""
## 🚀 FTTH Migration Opportunity

### Potential Impact:
- **Candidates:** {migration['data'].get('migration_candidates', 0):,}
- **Current Avg ARPU:** ₵{migration['data'].get('current_avg_arpu', 0):.0f}
- **Potential Avg ARPU:** ₵{migration['data'].get('potential_avg_arpu', 0):.0f}
- **Revenue Uplift:** ₵{migration['data'].get('revenue_uplift', 0):.0f}

**Recommendation:** {migration.get('recommendation', '')}
"""
    
    elif any(word in prompt_lower for word in ['demographic', 'age', 'gender', 'area']):
        age_results = json.loads(tools['demographics_by_age']())
        area_results = json.loads(tools['demographics_by_area']())
        
        # Generate visualizations
        age_df = pd.DataFrame(age_results['data']) if age_results['data'] else None
        if age_df is not None and not age_df.empty:
            age_df_renamed = age_df.copy()
            age_df_renamed = age_df_renamed.rename(columns={'age_group': 'Age Group'} if 'age_group' in age_df_renamed.columns else {})
            fig = ChartBuilder.build(age_df_renamed, ChartType.BAR,
                                    title="Demographics by Age")
            if fig:
                charts.append(("Age Demographics", fig))
        
        response = f"""
## 👥 Demographic Analysis

### By Age Group:
{_format_table(age_results['data'])}

### By Area:
{_format_table(area_results['data'])}

**Insights:** 
- {age_results.get('insight', '')}
- {area_results.get('insight', '')}
"""
    
    else:
        # Default: executive summary with dashboard
        dashboard = json.loads(tools['intelligence_dashboard']())
        
        # Create multi-panel visualization
        if 'panels' in dashboard and dashboard['panels']:
            for i, panel in enumerate(dashboard['panels'][:2]):  # Show first 2 panels
                panel_df = pd.DataFrame(panel.get('data', []))
                if not panel_df.empty:
                    viz_type = panel.get('visualization', {}).get('type', 'bar')
                    chart_type = ChartType[viz_type.upper()] if hasattr(ChartType, viz_type.upper()) else ChartType.BAR
                    fig = ChartBuilder.build(panel_df, chart_type, title=panel['title'])
                    if fig:
                        charts.append((panel['title'], fig))
        
        response = f"""
## 📊 Executive Summary

**Total Customers:** {dashboard['title']}

### Intelligence Dashboard Ready
{len(dashboard.get('panels', []))} analysis panels available for visualization
"""
    
    return response, charts


def _format_table(data: list) -> str:
    """Format list of dicts as markdown table"""
    if not data or not isinstance(data, list) or len(data) == 0:
        return "No data available"
    
    # Convert to DataFrame for cleaner formatting
    df = pd.DataFrame(data)
    return df.to_markdown(index=False)


if __name__ == "__main__":
    logger.info("🚀 Telecom Analytics Agent started")
