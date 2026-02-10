# Agno Agent Implementation Guide

## Why We Chose Agno

Agno is the perfect fit because:

1. **Simple Agent Creation**
```python
agent = Agent(
    model=OpenAIChat(...),
    tools=[...],
    instructions="Your prompt"
)
response = agent.run("Question")
```

2. **Excellent Tool Integration**
- Functions automatically become tools
- JSON response handling
- Parameter passing

3. **Local LLM Support**
- Works with Ollama, LM Studio, vLLM
- No cloud dependency
- Full control over inference

4. **Knowledge Base Integration**
- System prompt can include context
- Tools can access structured data
- Learnings can be persisted

---

## Core Agno Concepts

### 1. Agent
```python
agent = Agent(
    model=...,          # LLM to use
    tools=[...],        # Function tools
    description="...",  # What agent does
    instructions="...", # System prompt
    markdown=True       # Output format
)

response = agent.run(prompt)
```

### 2. Tools
```python
# Tools are functions that agent can call
def get_arpu_data():
    """Get ARPU analysis"""
    return {"metric": "ARPU", "data": [...]}

# Agent will automatically call when needed
```

### 3. Models
```python
# Local LLM
model = OpenAIChat(
    id="gpt-4",
    api_key="not-needed",
    base_url="http://192.168.120.227:7070/v1"
)

# Or cloud
model = OpenAIChat(id="gpt-4", api_key="sk-...")
```

### 4. Responses
```python
response = agent.run("Which technology has highest ARPU?")

# Response object has:
response.content      # AI response text
response.tools_used   # Which tools were called
response.metadata     # Additional info
```

---

## Our Implementation Architecture

### System Prompt Structure

The agent receives context through the system prompt:

```
Layer 1: Role Definition
├─ "You are a Giza telecom data analyst"
├─ "Your goal is to provide business intelligence"
└─ "Use tools to access data"

Layer 2: Tool Descriptions  
├─ "available_tools.arpu_analysis() - ARPU by dimension"
├─ "available_tools.churn_risk() - Risk assessment"
└─ "..."

Layer 3: Business Rules
├─ "ARPU is revenue divided by users"
├─ "Filter by latest insertion_date"
└─ "Dedup by subs_id"

Layer 4: Data Schemas
├─ "Table: analytic_models.Subscriber_Profile"
├─ "Columns: subs_id, ARPU, Current_Technology, ..."
└─ "..."

Layer 5: Known Patterns
├─ "FTTH has 40% higher ARPU"
├─ "Urban > Rural revenue"
└─ "Stable customers have 90% retention"
```

### Tool Routing Strategy

```
User Question
    │
    ▼
Parse Intent
    │
    ├─ Revenue/ARPU   → arpu_by_*() tools
    ├─ Risk/Churn     → churn_risk_*() tools  
    ├─ Growth/FTTH    → migration_*() tools
    ├─ Demographics   → demographics_*() tools
    └─ Other          → executive_summary()
    │
    ▼
Call Tool(s)
    │
    ▼
Format Response
    │
    ▼
Return to User
```

---

## Tool Function Patterns

### Pattern 1: Simple Data Tool
```python
def get_total_revenue() -> str:
    """Get total subscriber revenue"""
    query = "SELECT SUM(ARPU) as total FROM giza_data"
    result = duckdb.query(query)
    return json.dumps({"total": result['total'][0]})

# Agent calls: get_total_revenue()
# Returns: {"total": 1234567}
```

### Pattern 2: Multi-Row Analysis
```python
def arpu_by_technology() -> str:
    """ARPU breakdown by technology segment"""
    query = """
    SELECT Current_Technology, AVG(ARPU) as avg_arpu
    FROM giza_data
    GROUP BY 1
    ORDER BY 2 DESC
    """
    result = duckdb.query(query)
    return json.dumps(result.to_dict('records'))

# Returns: [
#   {"Current_Technology": "FTTH", "avg_arpu": 180},
#   {"Current_Technology": "LTE", "avg_arpu": 120}
# ]
```

### Pattern 3: Conditional Analysis
```python
def churn_risk(segment: str) -> str:
    """Analyze churn risk for segment"""
    query = f"""
    SELECT COUNT(*) as count, AVG(ARPU) as avg_arpu
    FROM giza_data
    WHERE Stability_Name = '{segment}'
    """
    result = duckdb.query(query)
    return json.dumps(result.to_dict('records')[0])

# Agent calls: churn_risk("At-Risk")
```

### Pattern 4: Complex Aggregation
```python
def migration_opportunity(tech_from: str, tech_to: str) -> str:
    """Calculate migration uplift"""
    query = f"""
    WITH current_seg AS (
        SELECT COUNT(*) as cnt, AVG(ARPU) as avg
        FROM giza_data
        WHERE Current_Technology = '{tech_from}'
    ),
    target_seg AS (
        SELECT AVG(ARPU) as avg
        FROM giza_data
        WHERE Current_Technology = '{tech_to}'
    )
    SELECT 
        cs.cnt,
        cs.avg as current_avg,
        ts.avg as target_avg,
        ts.avg - cs.avg as uplift_per_user
    FROM current_seg cs, target_seg ts
    """
    result = duckdb.query(query)
    return json.dumps(result.to_dict('records')[0])
```

---

## Request/Response Flow

### Example: "What's our top opportunity?"

```
REQUEST
───────
Question: "What's our top opportunity?"

AGENT PROCESSING  
─────────────────
1. Parse intent: Growth/Opportunity
2. Decide tools needed:
   - Get ARPU data
   - Check migration potential
   - Identify at-risk segments

3. Call tools:
   - arpu_by_technology() → result1
   - ftth_migration_analysis() → result2
   - at_risk_segments() → result3

4. Synthesize response:
   "Based on analysis:
    - FTTH currently 40% higher ARPU
    - 50,000 customers can migrate
    - Potential revenue uplift: ₵2.1M"

RESPONSE
────────
Formatted markdown response
with tables and recommendations
```

---

## Knowledge Base Integration

### Building Agent Context

```python
agent_context = f"""
You are a {company} telecom data analyst.

AVAILABLE DATA:
{schema_info}

BUSINESS RULES:
{business_rules}

PROVEN PATTERNS:
{query_patterns}

DISCOVERED INSIGHTS:
{learnings}

When answering questions:
1. Use available tools
2. Follow business rules
3. Reference proven patterns
4. Apply discovered insights
5. Provide metrics and recommendations
"""

agent = Agent(
    instructions=agent_context,
    tools=[...],
    model=...
)
```

---

## Error Handling in Tools

### Graceful Degradation

```python
def arpu_analysis() -> str:
    try:
        query = "SELECT ..."
        result = duckdb.query(query)
        if result is None or result.empty:
            return json.dumps({
                "error": "No data available",
                "fallback": "Using cached data from last run"
            })
        return json.dumps(result.to_dict('records'))
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "suggestion": "Try with broader filters"
        })
```

### Error Learning

```python
def log_error_pattern(question: str, error: str):
    """Log errors for knowledge base learning"""
    learnings = {
        "question": question,
        "error": error,
        "fix": "apply_this_fix()",
        "timestamp": datetime.now()
    }
    # Save to knowledge/learnings/errors.json
    # Later: agent learns to avoid this error
```

---

## Performance Optimization

### Tool Caching

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_arpu_by_technology_cached() -> str:
    """Cached version of expensive query"""
    # Called once per hour max
    # Returns stored result for same question
    return arpu_by_technology()
```

### Batch Tools

```python
def comprehensive_analysis() -> str:
    """One tool call returns multiple analyses"""
    return json.dumps({
        "arpu": get_arpu_data(),
        "churn": get_churn_data(),
        "migration": get_migration_data()
    })
    
# Better than 3 separate tool calls
```

### Streaming Responses

```python
response_stream = agent.run(
    prompt,
    stream=True
)

for chunk in response_stream:
    print(chunk, end='', flush=True)
    # User sees response as it's generated
```

---

## Advanced Patterns

### Pattern: Tool with State

```python
class DataAnalyzer:
    def __init__(self):
        self.last_query = None
        self.cached_result = None
    
    def analyze(self, question: str) -> str:
        if self.last_query == question:
            return self.cached_result
        
        result = self._run_analysis(question)
        self.last_query = question
        self.cached_result = result
        return result
```

### Pattern: Fallback Tools

```python
def get_metric(metric_name: str) -> str:
    """
    Try multiple data sources in order
    """
    # Try DuckDB first (fast)
    result = try_duckdb(metric_name)
    if result: return result
    
    # Fallback to Teradata (slower)  
    result = try_teradata(metric_name)
    if result: return result
    
    # Last resort: cached value
    return get_cached_metric(metric_name)
```

### Pattern: Conditional Tool Calling

```python
# Agent decides which tool to call based on context
if "by technology" in question:
    use_tool: arpu_by_technology
elif "by age" in question:
    use_tool: demographics_by_age  
else:
    use_tool: executive_summary
```

---

## Testing Your Agent

### Unit Test

```python
def test_arpu_tool():
    tools = create_tool_functions(duckdb_conn, km)
    result = json.loads(tools['arpu_by_technology']())
    
    assert 'data' in result
    assert len(result['data']) > 0
    assert 'Current_Technology' in result['data'][0]
```

### Integration Test

```python
def test_agent_flow():
    agent = Agent(
        model=ModelMock(),  # Mock model
        tools=tools,
        instructions="..."
    )
    
    response = agent.run("ARPU trends")
    assert "ARPU" in response.content
    assert len(response.tools_used) > 0
```

### End-to-End Test

```python
def test_full_pipeline():
    # Real Teradata + DuckDB + Agent
    teradata = TeradataConnector(...)
    teradata.connect()
    
    df = teradata.query(BASE_QUERY)
    duckdb.load_dataframe(df, "giza_data")
    
    agent = Agent(...)
    response = agent.run("Show me ARPU data")
    
    assert response.content
    assert "ARPU" in response.content or "Revenue" in response.content
```

---

## Deployment Considerations

### Streamlit Deployment

```python
# app_enhanced.py ready for:
# - Streamlit Cloud
# - Self-hosted Streamlit Server
# - Docker container
```

### Environment Variables

```
TERADATA_HOST=...
TERADATA_USER=...
TERADATA_PASSWORD=...
LLM_API_URL=...
LLM_MODEL=...
```

### Scaling

For more users:
- Increase DuckDB cache size
- Implement connection pooling
- Add API layer (FastAPI)
- Use background jobs

---

## Agno-Specific Debugging

### Enable Verbose Mode

```python
agent = Agent(
    ...,
    verbose=True  # Show all tool calls
)

response = agent.run("question")
# Output will show:
# Tool called: arpu_by_technology
# Result: [...]
# Agent reasoning: ...
```

### Access Tool Metadata

```python
response = agent.run("question")

print(response.tools_used)      # List of tool names
print(response.metadata)        # Execution details
print(response.tokens_used)     # If available
```

### Inspect Agent State

```python
response = agent.run("question")

# Check what agent "thought"
if hasattr(response, 'reasoning'):
    print(response.reasoning)
    
if hasattr(response, 'tool_calls'):
    for call in response.tool_calls:
        print(f"Called: {call.name}")
        print(f"Args: {call.args}")
        print(f"Result: {call.result}")
```

---

## Migration: From Tools to Agno

If you already have tool functions:

### Before (Raw functions)
```python
def analyze_arpu():
    result = duckdb.query("SELECT ...")
    return result

result = analyze_arpu()  # Direct call
```

### After (Agno tools)
```python
def analyze_arpu() -> str:
    """Analyze ARPU distribution"""  # Docstring = tool description
    result = duckdb.query("SELECT ...")
    return json.dumps(result.to_dict('records'))

agent = Agent(tools=[analyze_arpu])
response = agent.run("What's ARPU?")  # Agent calls the tool
```

**Only change:**
1. Return JSON strings (not objects)
2. Add docstrings  
3. Register with Agent

---

## Summary

**Agno gives you:**
- ✅ Simple agent creation
- ✅ Automatic tool integration
- ✅ Local LLM support
- ✅ Clean response handling
- ✅ Knowledge base integration
- ✅ Fast prototyping

**Start with the provided app_enhanced.py**
**Customize tools to your business**
**Scale with confidence**

Happy analyzing! 🚀

---

*Agno implementation guide - Built for your telecom use case*
*Version 1.0 - February 2026*
