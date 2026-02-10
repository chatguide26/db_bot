# 🧠 Telecom Analytics Intelligence Agent - Complete Guide

## Overview

This is an **AI-powered telecom analytics agent** that transforms raw Teradata subscriber data into instant business intelligence. It combines:

- **Teradata** - Enterprise data warehouse
- **DuckDB** - Ultra-fast in-memory analytics  
- **Local LLM** - On-premise intelligence
- **Agno Framework** - Agent orchestration
- **Layered Context** - Institutional knowledge
- **Interactive Visualizations** - Instant chart-based insights 📊 **NEW!**

## Architecture

```
Teradata Subscriber Data
        ↓
    [Extract]
        ↓
DuckDB (In-Memory)
        ↓
Knowledge Base Layers:
  ├── Table Schemas (knowledge/tables/)
  ├── Business Rules (knowledge/business/)
  ├── Query Patterns (knowledge/queries/)
  └── Learnings (knowledge/learnings/)
        ↓
    Agno Agent
        ├── Custom Tools (ARPU, Churn, Migration, Demographics)
        ├── Knowledge Manager (context aggregation)
        └── Tool Router (question → analysis)
        ↓
Streamlit Dashboard
  ├── Agent Chat Interface
  ├── Analytics Dashboard
  ├── Data Explorer
  └── Knowledge Base Viewer
```

## Technology Stack Comparison

### Option 1: **Agno** (Recommended for This Use Case) ✅

**Pros:**
- Built specifically for agent-based systems
- Excellent tool integration and function calling
- Native support for learning/knowledge persistence
- Works well with local LLMs
- Simple syntax, quick to prototype
- Good for multi-step reasoning chains

**Cons:**
- Less mature than LangChain
- Smaller community
- Limited production monitoring tools
- Fewer integrations out-of-the-box

**Best For:** Your use case - multi-tool orchestration, local LLM, knowledge base integration

### Option 2: **LlamaIndex + Agno** (Best for Knowledge Management) 🌟

**Pros:**
- LlamaIndex excels at knowledge base indexing
- Can index all your knowledge/ files automatically
- Built-in chunking, embedding, and retrieval
- Agno provides agent orchestration
- Best combination for structured knowledge

**Cons:**
- More complex setup
- Requires embedding models
- Higher memory footprint

**Best For:** If you want sophisticated RAG (Retrieval-Augmented Generation)

### Option 3: **LangChain** (Most Mature)

**Pros:**
- Mature, battle-tested
- Largest ecosystem & integrations  
- Excellent documentation
- Production-ready monitoring
- LangSmith integration for debugging

**Cons:**
- More verbose code
- Steeper learning curve
- Overkill for many use cases
- Can be slow

**Best For:** Large enterprise systems needing complex integrations

### Option 4: **CrewAI** (Multi-Agent Workflows)

**Pros:**
- Easy multi-agent collaboration
- Built-in role-based agents
- Good for complex workflows
- Cleaner than LangChain syntax

**Cons:**
- Newer, less stable
- Limited to simpler use cases
- Smaller community

**Best For:** If you need multiple specialized agents

## Our Recommendation: **Agno** ✅

We chose **Agno** for you because:

1. **Your workflow is agent-centric** - You need one smart agent with multiple tools
2. **Knowledge base is structured** - You have well-defined layers (tables, business, queries, learnings)
3. **Local LLM integration** - Agno works seamlessly with local models
4. **Simplicity** - Code is cleaner and easier to maintain
5. **Tool orchestration** - Perfect for routing questions to the right analysis

## File Structure

```
db_bot/
├── app.py                          # Original CSV-based app
├── app_enhanced.py                 # NEW: Agno + Teradata + Knowledge Base
├── requirements.txt                # Python dependencies
├── .env.example                    # Configuration template
│
├── src/
│   ├── connectors.py              # Teradata & DuckDB connectors
│   ├── knowledge_manager.py        # Load and manage knowledge base
│   └── agent_tools.py             # Custom tools for agent
│
└── knowledge/                      # Layered Context System
    ├── tables/
    │   └── subscriber_profile.json # Table schema + column definitions
    │
    ├── business/
    │   └── telecom_metrics.json    # KPIs, business rules, definitions
    │
    ├── queries/
    │   └── common_patterns.sql     # 8 verified SQL patterns
    │
    └── learnings/
        └── error_patterns.json     # Known errors, fixes, insights
```

## Key Features

### 1. **Layered Context System**

The agent has access to four layers of context:

```
Layer 1: Table Usage
  └─ Schemas, columns, relationships, sample queries
  └─ Source: knowledge/tables/subscriber_profile.json

Layer 2: Human Annotations  
  └─ Business metrics, definitions, thresholds
  └─ Source: knowledge/business/telecom_metrics.json

Layer 3: Query Patterns
  └─ 8 verified SQL patterns for common questions
  └─ Source: knowledge/queries/common_patterns.sql

Layer 4: Learnings
  └─ Error patterns, fixes, discovered insights
  └─ Source: knowledge/learnings/error_patterns.json
```

### 2. **Intelligent Tool Router**

The agent automatically routes questions:

```
"What are ARPU trends?"       → ARPU Analysis Tool
"Which customers are at risk?" → Churn Risk Tool
"How can we grow FTTH?"        → Technology Migration Tool
"What's our demographic profile?" → Demographics Tool
```

### 3. **Interactive Visualizations** 📊 NEW!

Every analysis now includes **auto-generated charts**:

```
Question: "What's ARPU by technology?"
├─ Response: Detailed analysis + metrics
├─ Chart: Bar chart (ARPU ranked by technology)
├─ Insight: "FTTH 40% higher than LTE"
└─ Action: Click to explore, zoom, download
```

**Chart Types:**
- **Bar** - Rankings, comparisons (ARPU by tech, geographic analysis)
- **Pie** - Distribution, composition (Risk segments, market share)
- **Line** - Trends over time (Growth trajectories, seasonal patterns)
- **Scatter** - Relationships, correlations (ARPU vs tenure)
- **Gauge** - KPIs, performance (Churn rate, achievement %)

**Features:**
- ✨ **Auto-detection** - Agent picks optimal chart type for data
- 🎨 **Interactive** - Hover, zoom, pan, download
- 📊 **Production quality** - Plotly professional charts
- 🔗 **Linked insights** - Charts explain themselves
- ⚡ **Real-time** - Generated in <100ms

### 4. **Data Quality Pipeline**

Automatic preprocessing:
- Deduplicates by subscriber (latest record)
- Standardizes technology names (Fiber → FTTH)
- Converts data types intelligently
- Removes critical nulls
- Handles outliers

### 5. **Knowledge Base Integration**

The agent can:
- Access table schemas automatically
- Explain any metric
- Reference known data quality issues
- Apply discovered insights
- Validate queries against business rules

## Running the Application

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment template
cp .env.example .env

# 3. Update .env with your Teradata credentials
# TERADATA_HOST=your_host
# TERADATA_USER=your_user
# TERADATA_PASSWORD=your_password
```

### Run CLI Version (Original)
```bash
streamlit run app.py
```

### Run Enhanced Version (Recommended)
```bash
streamlit run app_enhanced.py
```

Then:
1. Select governorate (Giza, Cairo, etc.)
2. Click "Initialize Agent"
3. Ask natural language questions:
   - "Which technology has highest ARPU?"
   - "Show me our at-risk customers"
   - "What's the FTTH migration opportunity?"
   - "Breakdown by age groups"

**NEW:** Each question returns **text analysis + interactive charts**!

## Visualization Experience 📊

When you ask a question, the agent provides:

1. **📝 AI Analysis** - Detailed text explanation with metrics
2. **📊 Interactive Charts** - Auto-selected visualization with hover/zoom/download
3. **💡 Business Insight** - What it means and recommended actions

**Example:**
```
You: "ARPU trends by technology?"

Agent returns:
─────────────────────────────────
📝 ## ARPU Analysis
   - FTTH: ₵180 avg, 12K subscribers (+40% vs LTE)
   - LTE: ₵120 avg, 25K subscribers
   - 3G: ₵85 avg, 8K subscribers

📊 [Interactive Bar Chart showing ARPU Rankings]

💡 KEY INSIGHT:
   "FTTH dominates profitability. Migrating 10K users could 
    yield ₵600K additional annual revenue."
```

See [**VISUALIZATIONS_GUIDE.md**](VISUALIZATIONS_GUIDE.md) for complete details.

## Example Queries

### ARPU Analysis
```
"What are our ARPU trends by technology?"
"Which segment generates most revenue?"
"How does FTTH ARPU compare to LTE?"
```

### Churn Risk
```
"How many customers are at risk?"
"What's the revenue at risk in each technology?"
"Which stable customers have churned risk?"
```

### Growth Opportunities  
```
"How many customers can migrate to FTTH?"
"What's the potential revenue from FTTH upgrades?"
"Which demographic segment should we target?"
```

### Segment Analysis
```
"What's the age profile of our customers?"
"Compare urban vs rural ARPU"
"How is tenure distributed?"
```

## Customization Guide

### Add New Metric

1. **Add to knowledge base** (`knowledge/business/telecom_metrics.json`):
```json
"MyMetric": {
  "name": "My Metric Name",
  "formula": "Calculation formula",
  "unit": "EGP"
}
```

2. **Create tool** (`src/agent_tools.py`):
```python
class MyTool(TelecomAnalyticsTool):
    def analyze(self):
        query = "SELECT ..."
        result = self.duckdb.query(query)
        return {"metric": "...", "data": ...}
```

3. **Register tool** in `create_tool_functions()`:
```python
"my_analysis": lambda: json.dumps(MyTool(...).analyze())
```

### Add New Query Pattern

1. Add to `knowledge/queries/common_patterns.sql`:
```sql
-- PATTERN 9: My New Pattern  
SELECT ...
```

2. Update `QueryBuilder.suggest_query_for_question()` in `knowledge_manager.py`

### Add New Data Source

1. Create new connector class in `src/connectors.py`
2. Load data to DuckDB: `duckdb.load_dataframe(df, "table_name")`
3. Update query patterns to include new table

## Performance Optimization

### DuckDB Caching
```python
# Query results are cached in memory
# Subsequent queries run in <100ms
```

### Index Key Columns
DuckDB automatically indexes:
- `subs_id` (subscriber identifier)
- `GOV` (governorate filter)
- `ARPU`, `Current_Technology` (analysis dimensions)

### Lazy Loading
- Only load selected governorate data
- Use column selection in explorer
- Filter before aggregation

## Monitoring & Logging

All agent operations logged:
```
✅ Connected to Teradata: 10.19.199.28/Tedata_temp
✅ Data preprocessing complete: 156,234 clean rows
✅ DuckDB query returned 45,000 rows
🤖 Question: "What are ARPU trends?"
📊 Response generated successfully
```

View logs in `streamlit_app.log`

## Troubleshooting

### "Teradata connection failed"
- Check TERADATA_HOST, credentials in .env
- Verify network connectivity
- Check firewall rules

### "No data returned"
- Verify governorate exists in database
- Check Subscriber_Status field values
- Try different date range

### "DuckDB query error"
- Check table name (must be `giza_data`)
- Verify column names match schema
- Use SELECT * to see actual columns

### "Knowledge base not loaded"
- Verify `knowledge/` directory exists
- Check JSON/SQL file syntax
- Review logs for specific errors

## Advanced: Agno Agent Configuration

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(
        id="gpt-4",
        base_url="http://192.168.120.227:7070/v1"  # Local LLM
    ),
    tools=[...],  # Your custom tool functions
    instructions=km.build_agent_context(),  # Knowledge base
    description="Telecom Data Analyst"
)

response = agent.run("What are ARPU trends?")
```

## Future Enhancements

1. **Multi-Agent Workflows** - Use CrewAI for specialized agents
   - Revenue Optimization Agent  
   - Churn Prevention Agent
   - Growth Strategy Agent

2. **Continuous Learning** - Auto-update knowledge base
   - Store successful query patterns
   - Log error fixes
   - Track insights

3. **Predictive Analytics** - Add ML models
   - Churn prediction
   - ARPU forecasting
   - Segment recommendations

4. **Real-time Dashboards** - Streaming updates
   - Live KPI feeds
   - Alert system
   - Anomaly detection

5. **Multi-Source Integration**
   - Add network management data
   - Integrate customer support tickets
   - Include billing data

## Support & Questions

For issues with:
- **Teradata**: Check connectivity, credentials, SQL syntax
- **DuckDB**: Verify data types, table names
- **Agno**: Review agent configuration, tool definitions
- **Streamlit**: Check session state, rerun() behavior

---

**Last Updated:** February 10, 2026
**Status:** Production Ready
**Version:** 1.0 Enhanced with Agno + Knowledge Base
