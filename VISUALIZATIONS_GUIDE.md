# 📊 Visualization Enhancement Guide

## Overview

The AI Data Analyst now generates **interactive visualizations alongside insights**. This turns Teradata data into immediate business intelligence with visual narratives.

## What Changed

### Before
- Agent returns: Text analysis with metrics
- User sees: Markdown tables and paragraphs
- Insight: Text-based only

### Now ✨
- Agent returns: **Text + Visualizations**
- User sees: Instant charts and graphs
- Insight: **Visual + textual narrative**

## Architecture

```
User Question
    │
    ▼
_route_question()
    │
    ├─ Identify intent (ARPU, Risk, Migration, Demographics)
    │
    ├─ Call analysis tools
    │   └─ Extract data as DataFrame
    │
    ├─ Generate visualizations
    │   ├─ ChartBuilder picks optimal chart type
    │   ├─ Plotly renders interactive chart
    │   └─ Return (text, [charts])
    │
    ▼
Streamlit UI
    ├─ Display: Text analysis (markdown)
    ├─ Display: Charts (Plotly figures)
    └─ User explores interactively
```

## Key Components

### 1. **VisualizationManager** (`src/visualizations.py`)

Orchestrates the visualization pipeline:

```python
# Suggests optimal chart type
chart_type = ChartDetector.detect(question, df)

# Builds interactive Plotly chart
fig = ChartBuilder.build(df, chart_type)

# Provides metadata and recommendations
metadata = create_visualization_metadata(question, df, analysis)
```

### 2. **ChartDetector**

Routes questions to chart types using keyword matching:

| Keyword | Chart Type |
|---------|-----------|
| distribution, share, composition | Pie |
| compare, vs, top, ranking | Bar |
| trend, over time, growth | Line |
| correlation, relationship | Scatter |
| risk, performance, kpi | Gauge |

**Example:**
```python
question = "What's ARPU by technology?"
chart_type = ChartDetector.detect(question, df)
# Returns: ChartType.BAR
```

### 3. **ChartBuilder**

Creates production-quality Plotly charts:

```python
# Detects x/y columns automatically
# Applies best practices (sorting, colors, labels)
# Returns interactive Plotly figure
fig = ChartBuilder.build(df, ChartType.BAR)
st.plotly_chart(fig)
```

### 4. **Enhanced Agent Tools**

Each analysis tool now includes visualization readiness:

```python
class ARPUAnalysisTool:
    def by_technology(self):
        result = {
            "metric": "ARPU by Technology",
            "data": [...],  # DataFrame-ready
            "visualization": {...}  # Chart metadata
        }
```

## Question-to-Chart Mapping

### ARPU Queries
```
Q: "What's ARPU by technology?"
┌─ Text: Technology breakdown table
├─ Chart: Bar chart (Ranked by ARPU)
└─ Insight: "FTTH dominates with X% higher ARPU"
```

### Risk Queries
```
Q: "How many customers are at risk?"
┌─ Text: Risk segment metrics
├─ Chart: Pie chart (Risk distribution)
└─ Insight: "X% in At-Risk segment"
```

### Demographic Queries
```
Q: "What's our age profile?"
┌─ Text: Age group table
├─ Chart: Bar chart (Age distribution)
└─ Insight: "25-34 segment is largest"
```

### Geography Queries
```
Q: "Compare governorates"
┌─ Text: Geographic ARPU table
├─ Chart: Bar chart (Gov comparison)
└─ Insight: "Urban significantly outperforms rural"
```

## Chart Types & Use Cases

### Bar Chart (Most Common)
```
Use for: Rankings, comparisons, breakdowns
Example: ARPU by Technology, Top Customers
When: 1 categorical + 1 numeric
Best practice: Sort descending, show values
```

### Pie Chart (Distribution)
```
Use for: Share of total, composition
Example: Stability distribution, Market share
When: 1 categorical + percentages
Best practice: Limit to 6-8 slices, show %
```

### Line Chart (Trends)
```
Use for: Time series, growth patterns
Example: ARPU trend, Subscriber growth
When: Date + metric columns
Best practice: Show markers, add range
```

### Scatter Plot (Relationships)
```
Use for: Correlations, outliers
Example: ARPU vs Tenure, Price vs Volume
When: 2+ numeric columns
Best practice: Add trend line, color by category
```

### Gauge Chart (KPIs)
```
Use for: Performance tracking
Example: Churn rate, Achievement %
When: Single metric with target
Best practice: Show target line, zones
```

## Code Examples

### Example 1: Detect Chart Type

```python
from src.visualizations import ChartDetector

question = "Top ARPU technologies"
df = subscriber_data

chart_type = ChartDetector.detect(question, df)
# Returns: ChartType.BAR
```

### Example 2: Build a Chart

```python
from src.visualizations import ChartBuilder, ChartType

df = arpu_by_tech_df
fig = ChartBuilder.build(
    df, 
    ChartType.BAR,
    title="ARPU by Technology",
    x_col="Current_Technology",
    y_col="avg_arpu"
)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)
```

### Example 3: Route with Visualizations

```python
# In app_enhanced.py
response, charts = _route_question(prompt, tools, km, duckdb)

# Display text
st.markdown(response)

# Display charts
for title, fig in charts:
    st.plotly_chart(fig)
```

## Visualization Knowledge Base

File: `knowledge/business/visualization_patterns.json`

Contains:
- Chart type→use case mapping
- Best practices per chart type
- When to use which visualization
- Color strategies
- Annotation guidelines
- AI narration templates

Example:
```json
{
  "distribution_visualization": {
    "use_cases": ["Market share", "Segment breakdown"],
    "keywords": ["distribution", "share", "composition"],
    "optimal_charts": ["pie", "donut", "sunburst"]
  }
}
```

## Interactive Features

### All Charts Include:
- ✅ **Hover**: Detailed info on mouseover
- ✅ **Zoom**: Explore specific regions
- ✅ **Pan**: Move around zoomed view
- ✅ **Download**: Save as PNG
- ✅ **Legend**: Toggle series on/off

### Streamlit Integration:
```python
# Fullwidth rendering
st.plotly_chart(fig, use_container_width=True)

# Multiple charts side-by-side
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1)
with col2:
    st.plotly_chart(fig2)

# Expandable sections
with st.expander("📊 Show detailed charts"):
    st.plotly_chart(fig)
```

## AI-Driven Insights

The visualization system enables AI to provide **visual + textual narratives**:

```
Chart shows: FTTH significantly higher ARPU
AI explains: "FTTH leads with ₵180 avg ARPU vs LTE ₵120"
AI recommends: "Accelerate FTTH migration for X% revenue uplift"
```

### Insight Templates

```python
# Pattern: Concentration risk
if pie_chart_top_3 > 70:
    insight = f"Revenue concentrated in top 3 segments ({concentration}%)"
    recommendation = "Diversify to reduce risk"

# Pattern: Growth opportunity
if trend_upward and segment_untapped:
    insight = f"{segment} growing {growth_rate}%"
    recommendation = "Expand investment in {segment}"

# Pattern: Performance gap
if max_arpu - min_arpu > threshold:
    insight = f"Performance gap: {gap_value} between top and bottom"
    recommendation = "Lift underperformers by {action}"
```

## Performance & Optimization

### Chart Rendering Speed
```
Chart type          | Render time  | Best for rows
─────────────────────────────────────────────
Bar/Pie            | <100ms       | <1000
Line (timeseries)  | <500ms       | <5000
Scatter            | <1000ms      | <2000
Heatmap            | <2000ms      | <500
```

### Data Density Rules
```
Rows < 10        → Simple table
Rows 10-500      → Standard chart
Rows 500-5000    → Aggregated chart
Rows > 5000      → GROUP BY + limit
```

## Customization Guide

### Add New Chart Type

```python
# 1. Add to ChartType enum
class ChartType(Enum):
    CUSTOM = "custom"

# 2. Implement builder
def _build_custom(df, title, x_col, y_col):
    fig = # Custom Plotly logic
    return fig

# 3. Register in build()
elif chart_type == ChartType.CUSTOM:
    return ChartBuilder._build_custom(...)

# 4. Add keyword mapping
'your_keyword': ChartType.CUSTOM
```

### Customize Chart Appearance

```python
# Modify ChartBuilder._build_bar()
fig.update_layout(
    height=500,  # Custom height
    colorway=['#FF6B6B', '#4ECDC4'],  # Colors
    font=dict(size=12),  # Font
    hovermode='x unified'  # Hover behavior
)
```

### Add Visualization Pattern

```python
# In knowledge/business/visualization_patterns.json
{
  "my_pattern": {
    "use_cases": [...],
    "keywords": [...],
    "optimal_charts": [...],
    "recommendation": "..."
  }
}
```

## Testing Visualizations

### Unit Test

```python
def test_chart_builder():
    df = pd.DataFrame({
        'Category': ['A', 'B', 'C'],
        'Value': [10, 20, 30]
    })
    fig = ChartBuilder.build(df, ChartType.BAR)
    assert fig is not None
    assert len(fig.data) > 0
```

### Integration Test

```python
def test_visualization_workflow():
    question = "ARPU by technology"
    response, charts = _route_question(
        question, tools, km, duckdb
    )
    assert len(charts) > 0
    assert response is not None
```

## Troubleshooting

### Chart Not Displaying

```
Problem: st.plotly_chart() returns blank
Solution: 
  1. Check DataFrame is not empty
  2. Verify column names exist
  3. Check for NaN values
```

### Wrong Chart Type Selected

```
Problem: Bar chart suggested for time series
Solution:
  1. Check keyword matching in ChartDetector
  2. Add 'trend' or 'time' to keywords
  3. Verify data has date column
```

### Performance Issues

```
Problem: Chart takes >2 seconds
Solution:
  1. Reduce rows (use LIMIT in SQL)
  2. Aggregate by GROUP BY
  3. Use appropriate chart type
  4. Check browser rendering
```

## Advanced: Custom Narration

Pair charts with AI-generated narratives:

```python
# Auto-generate insight from chart
insight = generate_insight(df, chart_type)
# Example output: "FTTH ARPU 40% higher than LTE"

st.markdown(f"### 💡 Key Insight\n{insight}")
st.plotly_chart(fig)
```

## Migration from CSV to Teradata

The visualization system works identically for both:

```python
# Before (CSV)
df = pd.read_csv('data.csv')
fig = ChartBuilder.build(df, ChartType.BAR)

# After (Teradata via DuckDB)
df = duckdb.query("SELECT ...").fetchdf()
fig = ChartBuilder.build(df, ChartType.BAR)
# Same visualization!
```

## Future Enhancements

```
Phase 1 (Now): ✅ Chart types, auto-detection, interactive
Phase 2: Add dashboard templates (KPI cards, metrics grid)
Phase 3: ML-driven chart selection (based on data patterns)
Phase 4: Multi-chart narratives (story-driven insights)
Phase 5: Collaborative dashboards (save/share analyses)
```

## Summary

The visualization enhancement makes the AI Data Analyst **10x more powerful** by:

| Before | After |
|--------|-------|
| Show numbers | Visualize patterns |
| Text-heavy | Visual-first |
| Manual creation | Auto-generated |
| Static tables | Interactive charts |
| Hard to compare | Instant comparison |

**Result:** Users get **instant business intelligence** with interactive charts + AI insights.

---

*Visualization Enhancement Guide - Turn data into instant insights*
*Version 1.0 - February 2026*
