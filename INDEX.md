# 📚 Documentation Index

Welcome to the Telecom Analytics Intelligence Agent! This guide helps you navigate all documentation.

## 🎯 Start Here

1. **[LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)** 🖥️ - **Setup on Your Machine (NEW!)**
   - Clone project to local VS Code
   - Set up Python virtual environment
   - Configure local LLM (Ollama, LM Studio)
   - Full troubleshooting guide
   - Perfect for development!

2. **[QUICKSTART.md](QUICKSTART.md)** ⚡ - **5 minutes to running**
   - Installation & dependencies
   - First query examples
   - Troubleshooting

3. **[README.md](README.md)** 📖 - **Complete guide**
   - Full architecture explanation
   - Feature overview
   - File structure
   - Running instructions
   - Customization guide

4. **[VISUALIZATIONS_GUIDE.md](VISUALIZATIONS_GUIDE.md)** 📊 - **Interactive charts**
   - What's new with visualizations
   - Chart type selection
   - AI-driven visualization insights
   - Customization examples

## 🏗️ Architecture & Design

3. **[TECH_COMPARISON.md](TECH_COMPARISON.md)** 🔧 - **Technology decisions**
   - Why we chose Agno
   - Comparison with LangChain, LlamaIndex, CrewAI
   - Decision matrix
   - Migration paths
   - Cost analysis

4. **[AGNO_GUIDE.md](AGNO_GUIDE.md)** 🤖 - **Agno implementation details**
   - Core concepts
   - Tool patterns
   - Request/response flow
   - Knowledge integration
   - Advanced patterns
   - Testing & deployment

## 📂 Source Code

### Main Application
- **[app_enhanced.py](app_enhanced.py)** - Enhanced Streamlit app with Agno
- **[app.py](app.py)** - Original CSV-based app

### Core Modules (src/)
- **[src/connectors.py](src/connectors.py)** - Teradata & DuckDB connectors
- **[src/knowledge_manager.py](src/knowledge_manager.py)** - Knowledge base management
- **[src/agent_tools.py](src/agent_tools.py)** - Custom analysis & visualization tools
- **[src/visualizations.py](src/visualizations.py)** - **NEW:** Chart generation & AI-driven visualization
- **[src/__init__.py](src/__init__.py)** - Package exports

## 📚 Knowledge Base

The `knowledge/` directory contains 4 layers of institutional knowledge:

### Layer 1: Table Schemas
- **[knowledge/tables/subscriber_profile.json](knowledge/tables/subscriber_profile.json)**
  - Teradata table definition
  - Column descriptions & types
  - Available aggregations
  - Sample queries

### Layer 2: Business Rules  
- **[knowledge/business/telecom_metrics.json](knowledge/business/telecom_metrics.json)**
  - ARPU definition & thresholds
  - Churn risk indicators
  - Technology migration rules
  - Data quality checks
  - Reporting standards

### Layer 3: Query Patterns
- **[knowledge/queries/common_patterns.sql](knowledge/queries/common_patterns.sql)**
  - 8 proven SQL patterns
  - Use cases for each pattern
  - Expected output
  - Performance notes

### Layer 4: Learnings
- **[knowledge/learnings/error_patterns.json](knowledge/learnings/error_patterns.json)**
  - Known data issues & fixes
  - Query optimizations
  - Discovered business insights
  - Recommendations

## ⚙️ Configuration

- **[requirements.txt](requirements.txt)** - Python dependencies
- **[.env.example](.env.example)** - Configuration template

## 🚀 Quick Navigation

### I want to...

**...set up locally on my machine**
→ [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) (Complete setup + local LLM guide)

**...get started immediately**
→ [QUICKSTART.md](QUICKSTART.md) (5 min)

**...understand the full system**  
→ [README.md](README.md) (30 min)

**...know why we chose Agno**
→ [TECH_COMPARISON.md](TECH_COMPARISON.md) (15 min)

**...implement a custom tool**
→ [AGNO_GUIDE.md](AGNO_GUIDE.md) - "Advanced Patterns" section

**...add or customize visualizations**
→ [VISUALIZATIONS_GUIDE.md](VISUALIZATIONS_GUIDE.md) - "Customization" section

**...add new business metrics**
→ [README.md](README.md) - "Customization Guide" section

**...troubleshoot an issue**
→ [README.md](README.md) - "Troubleshooting" section
→ [QUICKSTART.md](QUICKSTART.md) - "Troubleshooting" section

**...understand the architecture**
→ [README.md](README.md) - "Architecture" section  
→ [AGNO_GUIDE.md](AGNO_GUIDE.md) - "Architecture" section

**...compare frameworks**
→ [TECH_COMPARISON.md](TECH_COMPARISON.md) (comprehensive)

## 📊 Feature Overview

| Feature | File | Documentation |
|---------|------|-----------------|
| Teradata Connection | src/connectors.py | README.md |
| DuckDB Analytics | src/connectors.py | README.md |
| Data Preprocessing | src/connectors.py | README.md |
| Knowledge Base | src/knowledge_manager.py | README.md |
| ARPU Analysis | src/agent_tools.py | AGNO_GUIDE.md |
| Churn Risk | src/agent_tools.py | AGNO_GUIDE.md |
| FTTH Migration | src/agent_tools.py | AGNO_GUIDE.md |
| Demographics | src/agent_tools.py | AGNO_GUIDE.md |
| Chart Detection | src/visualizations.py | VISUALIZATIONS_GUIDE.md |
| Chart Building | src/visualizations.py | VISUALIZATIONS_GUIDE.md |
| AI Visualization Insights | src/visualizations.py | VISUALIZATIONS_GUIDE.md |
| Agent Routing | app_enhanced.py | AGNO_GUIDE.md |
| Streamlit UI | app_enhanced.py | README.md |
| Interactive Dashboards | app_enhanced.py | VISUALIZATIONS_GUIDE.md |

## 📈 Learning Path

### Local Development Setup (1-2 hours)
1. Start with [CHEATSHEET.md](CHEATSHEET.md) for 30-second overview
2. Follow [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for detailed steps
3. Clone project to local machine
4. Set up Python virtual environment
5. Install Ollama or LM Studio
6. Run `streamlit run app_enhanced.py`
7. Test with sample questions

### Beginner (1-2 hours)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run the application
3. Try basic queries
4. Explore each tab

### Intermediate (4-6 hours)  
1. Read [README.md](README.md)
2. Understand architecture
3. Examine source code
4. Customize metrics
5. Add new query patterns

### Advanced (8+ hours)
1. Study [TECH_COMPARISON.md](TECH_COMPARISON.md)
2. Master [AGNO_GUIDE.md](AGNO_GUIDE.md)
3. Build custom tools
4. Implement learnings persistenceé
5. Design multi-agent system

## 📘 Documentation by Role

### Business Analyst
- [QUICKSTART.md](QUICKSTART.md) - How to use
- [README.md](README.md) - Features & examples
- knowledge/ files - Understand business rules

### Data Analyst
- [README.md](README.md) - Full guide
- [src/connectors.py](src/connectors.py) - Data sources
- [knowledge/queries/](knowledge/queries/) - SQL patterns

### Engineer
- [AGNO_GUIDE.md](AGNO_GUIDE.md) - Implementation details
- [src/](src/) - Source code
- [TECH_COMPARISON.md](TECH_COMPARISON.md) - Architecture decisions

### DevOps/Platform
- [README.md](README.md) - Deployment section
- [requirements.txt](requirements.txt) - Dependencies
- [.env.example](.env.example) - Configuration

## 🔍 Key Concepts Glossary

| Term | Definition | Where |
|------|-----------|-------|
| **Agno** | Agent framework for tool orchestration | AGNO_GUIDE.md |
| **Knowledge Base** | 4 layers of institutional context | AGNO_GUIDE.md |
| **DuckDB** | In-memory analytics engine | README.md |
| **ARPU** | Average Revenue Per User (key metric) | knowledge/business/ |
| **Churn Risk** | Probability customer will leave | knowledge/business/ |
| **Tool** | Function agent can call | AGNO_GUIDE.md |
| **Query Pattern** | Proven SQL template | knowledge/queries/ |
| **Learning** | Discovered insight or error fix | knowledge/learnings/ |
| **Chart Detection** | AI identifies optimal visualization type | VISUALIZATIONS_GUIDE.md |
| **Visualization Pattern** | Knowledge base for chart selection | knowledge/business/ |
| **Insight Generation** | AI explains trends & patterns in charts | VISUALIZATIONS_GUIDE.md |

## 🔗 External Resources

- [Agno GitHub](https://github.com/prefix-dev/agno)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Teradata SQL Guide](https://support.teradata.com/)

## 📞 Support

For issues related to:
- **Agno**: See [AGNO_GUIDE.md](AGNO_GUIDE.md) Debugging section
- **Setup**: See [QUICKSTART.md](QUICKSTART.md) Troubleshooting
- **Architecture**: See [README.md](README.md)
- **Technology choice**: See [TECH_COMPARISON.md](TECH_COMPARISON.md)

## 📝 Files Summary

```
Total Files: 23
├── Documentation: 8 files (README, TECH_COMPARISON, AGNO_GUIDE, QUICKSTART, 
│                           VISUALIZATIONS_GUIDE, LOCAL_SETUP_GUIDE, CHEATSHEET, INDEX)
├── Source Code: 7 files (app, visualizations, src modules)
├── Knowledge Base: 5 files (tables, business with visualization patterns, queries, learnings)
├── Config: 2 files (requirements, .env.example)
└── Other: 1 file (DATA_COLLECTION_SYNC.md for future)
```

## ✅ Checklist

- [ ] Read QUICKSTART.md
- [ ] Install requirements
- [ ] Run app_enhanced.py
- [ ] Try 3 different questions
- [ ] Read README.md
- [ ] Understand architecture
- [ ] Explore source code
- [ ] Customize for your use case
- [ ] Deploy to production

---

## Quick Links

⚡ **[Cheatsheet: CHEATSHEET.md](CHEATSHEET.md)** ← 30-second setup
🖥️ **[Local Setup: LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)** ← Full setup guide
🚀 **[Quick Start: QUICKSTART.md](QUICKSTART.md)**
📖 **[Full Docs: README.md](README.md)**  
📊 **[Visualizations: VISUALIZATIONS_GUIDE.md](VISUALIZATIONS_GUIDE.md)**
🤖 **[Agno Details: AGNO_GUIDE.md](AGNO_GUIDE.md)**
🔧 **[Tech Choice: TECH_COMPARISON.md](TECH_COMPARISON.md)**

---

*Documentation Index - Navigate the complete system*
*Last updated: February 10, 2026*
