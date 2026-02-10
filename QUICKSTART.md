# 🚀 Quick Start Guide - 5 Minutes to Running Agent

## Prerequisites

- Python 3.10+
- Teradata access (host 10.19.199.28)
- Local LLM running (port 7070) or modify for OpenAI

## Installation (2 minutes)

```bash
# Clone/navigate to project
cd db_bot

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Teradata credentials
```

## Run Application (1 minute)

```bash
# Enhanced version with Agno integration (RECOMMENDED)
streamlit run app_enhanced.py

# Or original CSV version
streamlit run app.py
```

## First Query (2 minutes)

1. Select governorate: **Giza**
2. Click **"Initialize Agent"**
3. Wait for data load (~30 seconds)
4. Ask a question:

### Top Queries to Try

**ARPU Analysis**
```
"What's our average ARPU by technology?"
"Which segment generates most revenue?"
"Show ARPU trends"
```

**Churn Risk**
```
"How many customers are at risk?"
"Show at-risk revenue by technology"
"What's our churn rate?"
```

**Growth Opportunities**
```
"How many can migrate to FTTH?"
"What's the FTTH revenue uplift?"
"Calculate growth potential"
```

**Demographics**
```
"Break down by age groups"
"Compare urban vs rural"
"Show tenure distribution"
```

---

## Architecture at a Glance

```
You Ask Question
        │
        ▼
    Agno Agent
    (Router)
        │
    ┌───┴───┬───────┬──────────┐
    │       │       │          │
    ▼       ▼       ▼          ▼
 ARPU    Churn  Migration  Demographics
 Tool    Tool    Tool       Tool
    │       │       │          │
    └───────┴───────┴──────────┘
            │
            ▼
       DuckDB
    (Fast Analytics)
            │
            ▼
   Knowledge Base
   (4 Context Layers)
            │
            ▼
    Structured Response
```

---

## File Structure

```
db_bot/
├── app_enhanced.py           ◄─── RUN THIS
├── src/
│   ├── connectors.py        ◄─── Teradata/DuckDB
│   ├── knowledge_manager.py  ◄─── Knowledge base
│   └── agent_tools.py       ◄─── Analysis tools
├── knowledge/
│   ├── tables/              ◄─── Schema definitions
│   ├── business/            ◄─── Business rules
│   ├── queries/             ◄─── SQL patterns
│   └── learnings/           ◄─── Discovered insights
└── requirements.txt
```

---

## Key Concepts

### 1. Layered Context  
Agent has access to:
- **Tables**: Column definitions, types, relationships
- **Business**: KPI definitions, thresholds, rules
- **Queries**: 8 proven SQL patterns
- **Learnings**: Error patterns, insights

### 2. Smart Tools
Agent automatically chooses:
- ARPU Analysis for revenue questions
- Churn Risk for customer risk
- FTTH Migration for growth
- Demographics for audience

### 3. Data Quality
Automatic preprocessing:
- Deduplicates records
- Fixes data types  
- Standardizes categories
- Removes bad data

### 4. DuckDB Speed
In-memory analytics:
- <100ms query time
- Supports complex aggregations
- Perfect for real-time questions

---

## Common Operations

### Change Data Source
```python
# In app_enhanced.py, modify BASE_QUERY
BASE_QUERY = """
    SELECT ...
    FROM your_table
    WHERE your_filters
"""
```

### Add New Question Type
```python
# In _route_question() function, app_enhanced.py
elif any(word in prompt_lower for word in ['your_keyword']):
    results = json.loads(tools['your_tool']())
    response = f"Your formatted response: {results}"
```

### Add New Analysis Tool
```python
# 1. Create class in src/agent_tools.py
class MyTool(TelecomAnalyticsTool):
    def analyze(self):
        query = "SELECT ..."
        return self.duckdb.query(query)

# 2. Register in create_tool_functions()
"my_tool": lambda: json.dumps(MyTool(...).analyze())

# 3. Reference in _route_question()
```

---

## Troubleshooting

### Teradata Connection Error
```
❌ Error: "host unreachable"

Solution:
1. Check ping: ping 10.19.199.28
2. Verify credentials in .env
3. Check firewall/VPN
```

### No Data Loaded
```
❌ Error: "No data returned"

Solution:
1. Check governorate exists
2. Verify Subscriber_Status column
3. Try SELECT COUNT(*) first
```

### LLM Not Responding
```
❌ Error: "Connection to 192.168.120.227:7070 failed"

Solution:
1. Start LLM: python -m llama_cpp.server --model model.gguf
2. Check port: curl http://localhost:7070/health
3. Verify base_url in .env
```

### DuckDB Query Error
```
❌ Error: "Unknown table: giza_data"

Solution:
1. Check data was loaded (click Initialize Agent)
2. Verify table name in connectors.py
3. Check column names match schema
```

---

## Next Steps

### 1️⃣ Get Familiar (Today)
- [ ] Run the app
- [ ] Try 5 different questions
- [ ] Check different tabs (Charts, Data Explorer, Knowledge Base)

### 2️⃣ Customize (Week 1)
- [ ] Add your business metrics
- [ ] Update knowledge base for your telco
- [ ] Create custom query patterns
- [ ] Adjust Streamlit UI

### 3️⃣ Integrate Data Sources (Week 2)
- [ ] Add network management data
- [ ] Include customer support tickets
- [ ] Connect billing system
- [ ] Pull real-time KPIs

### 4️⃣ Scale (Month 2)
- [ ] Add LlamaIndex for semantic search
- [ ] Implement learning persistence
- [ ] Build multi-agent system
- [ ] Deploy to production

---

## Pro Tips 💡

### 1. Ask Follow-Up Questions
```
Q: "What's our ARPU?"
A: [Response with numbers]
Q: "Why is FTTH higher?"  ◄─── Follow-up works!
A: [Detailed explanation]
```

### 2. Request Specific Formats
```
"Give me ARPU by technology as a table"
"Show demographics in JSON format"
"List top 10 at-risk customers"  
```

### 3. Combine Analyses
```
"Compare FTTH ARPU for urban customers over 30 years old"
"Which at-risk high-ARPU customers should we target?"
```

### 4. Monitor Knowledge Base
```
- Check discovery of new insights
- Review error patterns section
- Update business rules
```

---

## Production Checklist

Before deploying:

- [ ] Credentials in secure vault (not .env)
- [ ] Logging configured for monitoring
- [ ] Cache enabled for performance
- [ ] Error handling on failed queries
- [ ] Knowledge base versioned
- [ ] Backup of learnings.json
- [ ] Documentation updated
- [ ] Team trained on queries

---

## Performance Tips

**Speed up queries:**
- Filter by GOV first (indexed)
- Select specific columns  
- Use LIMIT for exploration
- Cache common analyses

**Optimize memory:**
- Load one governorate at a time
- Close DuckDB connection after use
- Clear chat history periodically

**Better responses:**
- Give agent specific context
- Ask for structured output
- Provide date ranges
- Specify required dimensions

---

## Support Resources

- 📖 [README.md](README.md) - Full documentation
- 🔧 [TECH_COMPARISON.md](TECH_COMPARISON.md) - Technology choices
- 📊 [knowledge/](knowledge/) - Reference data
- 🐛 [GitHub Issues](https://github.com/your/repo/issues)

---

## That's It!

You now have a working AI agent that:

✅ Connects to Teradata  
✅ Analyzes data with DuckDB  
✅ Answers natural language questions  
✅ Provides structured intelligence  
✅ Learns from patterns  

**Next 5 steps:**
1. Initialize Agent
2. Ask about ARPU
3. Check Charts tab
4. Explore Data
5. Review Knowledge Base

Happy analyzing! 🚀

---

*Quick start guide - Get running in 5 minutes*
*For detailed info, see README.md*
