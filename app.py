import os
import json
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import csv
import hashlib
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools
import openai
import httpx

os.environ['NO_PROXY'] = '192.168.120.180,localhost,127.0.0.1,.local'

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ========== SIMPLE JSON EXTRACTOR ==========
def extract_json_from_response(content):
    """Extract JSON array from agent response"""
    # Method 1: Look for JSON array pattern
    json_pattern = r'\[\s*\{.*?\}\s*\]'
    matches = re.findall(json_pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # Clean up the match
            clean_match = match.strip()
            # Parse JSON
            data = json.loads(clean_match)
            if isinstance(data, list) and len(data) > 0:
                return data
        except json.JSONDecodeError:
            # Try with single quotes
            try:
                clean_match = clean_match.replace("'", '"')
                data = json.loads(clean_match)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except:
                continue
    return None

# ========== CHART CREATOR ==========
def create_chart_from_data(data, chart_type):
    """Create chart from extracted data"""
    if not data or len(data) == 0:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find category and value columns
        category_col = None
        value_col = None
        
        # Look for category column
        for col in df.columns:
            if 'category' in col or 'name' in col or 'label' in col:
                category_col = col
                break
        
        # Look for value column
        for col in df.columns:
            if 'value' in col or 'sales' in col or 'count' in col or 'total' in col:
                value_col = col
                break
        
        # Fallback: use first string column as category, first numeric as value
        if not category_col:
            for col in df.columns:
                if df[col].dtype == 'object':
                    category_col = col
                    break
        
        if not value_col:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col
                    break
        
        if not category_col or not value_col:
            return None
        
        # Convert value to numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        
        if len(df) == 0:
            return None
        
        # Create chart based on type
        if chart_type == 'pie':
            fig = px.pie(
                df, 
                names=category_col, 
                values=value_col,
                title=f"{category_col.title()} Distribution",
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
        
        elif chart_type == 'bar':
            # Sort by value
            df = df.sort_values(value_col, ascending=False)
            fig = px.bar(
                df,
                x=category_col,
                y=value_col,
                title=f"{category_col.title()} by {value_col.title()}",
                color=value_col,
                text=value_col
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
        
        elif chart_type == 'scatter':
            # For scatter, we need x and y columns
            if len(df.columns) >= 2:
                x_col = category_col if category_col != value_col else df.columns[0]
                y_col = value_col
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col.title()} vs {x_col.title()}",
                    size=y_col if len(df) < 20 else None
                )
        
        elif chart_type == 'line':
            # Check if we have a date-like column
            date_col = None
            for col in df.columns:
                if 'date' in col or 'time' in col or 'month' in col:
                    date_col = col
                    break
            
            if date_col:
                df = df.sort_values(date_col)
                fig = px.line(
                    df,
                    x=date_col,
                    y=value_col,
                    title=f"{value_col.title()} Trend"
                )
            else:
                fig = px.line(
                    df,
                    x=category_col,
                    y=value_col,
                    title=f"{value_col.title()} by {category_col.title()}"
                )
        
        else:
            # Default to bar chart
            df = df.sort_values(value_col, ascending=False)
            fig = px.bar(
                df,
                x=category_col,
                y=value_col,
                title=f"{category_col.title()} Distribution"
            )
        
        fig.update_layout(height=400, showlegend=True)
        return fig, df
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return None, None

# ========== DETECT CHART TYPE ==========
def detect_chart_type(query):
    query_lower = query.lower()
    if 'pie' in query_lower or 'distribution' in query_lower or 'share' in query_lower:
        return 'pie'
    elif 'bar' in query_lower or 'compare' in query_lower or 'top' in query_lower:
        return 'bar'
    elif 'line' in query_lower or 'trend' in query_lower or 'over time' in query_lower:
        return 'line'
    elif 'scatter' in query_lower or 'correlation' in query_lower or ' vs ' in query_lower:
        return 'scatter'
    return 'bar'

# ========== APP UI ==========
st.set_page_config(
    page_title="Chart-First Business Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Chart-First Business Analyst")
st.markdown("**Visualizations guaranteed from JSON data**")

# Sidebar
with st.sidebar:
    st.header("⚡ Configuration")
    
    # Chart display toggle
    auto_show_charts = st.toggle("Auto-show charts", True, 
                                help="Automatically display charts when JSON data is found")
    
    st.markdown("---")
    st.header("🎯 Chart Commands")
    
    st.markdown("**Try these exact phrases:**")
    st.code("Category distribution pie")
    st.code("Top categories bar chart")
    st.code("Price vs quantity scatter")
    st.code("Monthly trend line")
    
    st.markdown("---")
    st.header("📊 Chart Format")
    
    st.markdown("**Agent should output:**")
    st.code("""[
  {"category": "Fitness", "value": 28},
  {"category": "Electronics", "value": 17}
]""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Process file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        st.session_state.df = df
        
        # Show data info
        st.subheader("📋 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_cols)
        
        # Initialize agent
        if st.session_state.agent is None:
            with st.spinner("Initializing agent..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                    temp_path = temp_file.name
                    df.to_csv(temp_path, index=False)
                
                # Initialize DuckDB
                duckdb_tools = DuckDbTools()
                duckdb_tools.load_local_csv_to_table(temp_path, "uploaded_data")
                
                # Create agent with specific instructions for charts
                st.session_state.agent = Agent(
                    model=OpenAIChat(
                        id="gpt-4",
                        api_key="not-needed",
                        base_url="http://192.168.120.180:7070/v1",
                        temperature=0.1,
                        max_tokens=1000
                    ),
                    tools=[duckdb_tools, PandasTools()],
                    description="Data Analyst",
                    instructions="""You are a data analyst. Follow these rules STRICTLY:

1. ALWAYS include structured JSON data for charts at the END of your response
2. JSON format MUST be: [{"category": "Name", "value": 123}, ...]
3. Keep the JSON clean and parseable
4. Provide brief insights above the JSON
5. For pie charts: category distribution
6. For bar charts: top categories comparison
7. For scatter: x and y values
8. For line: time series data

Example response:
The data shows Fitness is most popular.

[
  {"category": "Fitness", "value": 28},
  {"category": "Electronics", "value": 17}
]""",
                    markdown=True
                )
        
        # Conversation display
        st.subheader("💬 Conversation")
        
        for msg in st.session_state.conversation[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Check if this message has JSON for chart
                if msg["role"] == "assistant" and auto_show_charts:
                    json_data = extract_json_from_response(msg["content"])
                    if json_data:
                        # Try to determine chart type from conversation context
                        chart_type = 'bar'  # default
                        if len(st.session_state.conversation) > 1:
                            last_user_msg = None
                            for prev_msg in reversed(st.session_state.conversation):
                                if prev_msg["role"] == "user":
                                    last_user_msg = prev_msg["content"]
                                    break
                            if last_user_msg:
                                chart_type = detect_chart_type(last_user_msg)
                        
                        fig, chart_df = create_chart_from_data(json_data, chart_type)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            with st.expander("📊 View chart data"):
                                st.dataframe(chart_df)
        
        # Chat input
        if prompt := st.chat_input("Ask for analysis or charts..."):
            # Add to conversation
            st.session_state.conversation.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        # Get agent response
                        response = st.session_state.agent.run(prompt)
                        content = response.content if hasattr(response, 'content') else str(response)
                        
                        # Display response
                        st.markdown(content)
                        
                        # Check for JSON and create chart
                        if auto_show_charts:
                            json_data = extract_json_from_response(content)
                            if json_data:
                                chart_type = detect_chart_type(prompt)
                                fig, chart_df = create_chart_from_data(json_data, chart_type)
                                
                                if fig:
                                    st.success("✅ Chart generated from JSON data!")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show the data used
                                    with st.expander("📋 View chart data"):
                                        st.dataframe(chart_df)
                                else:
                                    st.info("📊 JSON found but couldn't create chart")
                            else:
                                st.info("📊 No JSON data found for chart")
                        
                        # Add to conversation
                        st.session_state.conversation.append({"role": "assistant", "content": content})
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.conversation.append({"role": "assistant", "content": error_msg})
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

else:
    # Welcome screen
    st.markdown("""
    # 📊 Guaranteed Charts from JSON
    
    This version **WILL** show charts when the agent provides JSON data.
    
    ## 🎯 How it works:
    
    1. **Upload** your CSV/Excel file
    2. **Ask** for analysis with chart keywords:
       - "Category distribution pie"
       - "Top categories bar chart"
       - "Price vs quantity scatter"
    
    3. **Agent responds** with JSON like:
    ```json
    [
      {"category": "Fitness", "value": 28},
      {"category": "Electronics", "value": 17}
    ]
    ```
    
    4. **Chart appears automatically** below the response
    
    ## 🔧 Key Features:
    
    - **Simple JSON extraction** - looks for `[{...}, {...}]` patterns
    - **Automatic chart detection** - from query keywords
    - **Guaranteed display** - if JSON exists, chart shows
    - **Clean visualization** - with Plotly interactive charts
    
    ## 📝 Example Queries:
    
    - "Show category distribution as pie chart"
    - "Create bar chart of top 5 products"
    - "Visualize price vs quantity correlation"
    - "Monthly sales trend line chart"
    
    ---
    
    *Upload a file to see charts in action!*
    """)
