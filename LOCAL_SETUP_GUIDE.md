# 🖥️ Local Development Setup Guide

Run the Telecom Analytics Agent on your local VS Code with a local LLM. This guide covers full local development.

## 📋 Prerequisites

- **Python 3.10 or higher** (check: `python --version`)
- **Local LLM** (Ollama, LM Studio, Text Generation WebUI, or any OpenAI-compatible endpoint)
- **Git** (for cloning the repository)
- **VS Code** with Python extension installed
- **4GB+ RAM** (recommended for local LLM + DuckDB)

---

## 🚀 Step 1: Clone Project Locally

### Option A: Using Git
```bash
# Clone the repository
git clone https://github.com/chatguide26/db_bot.git
cd db_bot

# Or if using SSH:
git clone git@github.com:chatguide26/db_bot.git
cd db_bot
```

### Option B: Download as ZIP
1. Go to: https://github.com/chatguide26/db_bot
2. Click **Code → Download ZIP**
3. Extract to your preferred location
4. Open folder in VS Code: `code db_bot`

---

## 🐍 Step 2: Set Up Python Environment

### On Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### On macOS/Linux (Bash)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

**Verify activation** - you should see `(venv)` prefix in terminal

---

## 📦 Step 3: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; import pandas; import duckdb; print('✅ All dependencies installed!')"
```

### Troubleshooting Dependencies
If `teradatasql` fails to install (common on macOS/Linux):
```bash
# It's only needed if connecting to actual Teradata
# For local testing, this is optional - edit requirements.txt to remove it
pip install -r requirements.txt --ignore-installed teradatasql
```

---

## 🤖 Step 4: Set Up Local LLM (3 Options)

### Option A: Ollama (Recommended - Easiest)

1. **Download & Install Ollama**
   - Windows/macOS: https://ollama.ai
   - Linux: `curl https://ollama.ai/install.sh | sh`

2. **Pull a Model** (Choose one):
   ```bash
   # Small & fast (2GB)
   ollama pull mistral
   
   # Balanced (7GB)
   ollama pull llama2
   
   # Powerful but slower (13GB)
   ollama pull neural-chat
   ```

3. **Start Ollama Server** (automatic on Windows/macOS, manual on Linux):
   ```bash
   ollama serve
   # Runs on: http://localhost:11434
   ```

4. **Test it works**:
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"mistral","prompt":"hello"}'
   ```

### Option B: LM Studio (GUI-based)

1. Download from: https://lmstudio.ai
2. Launch LM Studio → Search "mistral" → Download
3. Click **"Start Server"** button
4. Server runs on: `http://localhost:1234`

### Option C: Text Generation WebUI

1. Clone: `git clone https://github.com/oobabooga/text-generation-webui`
2. Follow their setup (Python venv required)
3. Download model via UI
4. Start with GPU acceleration (optional)
5. Enable API in "Parameters" tab

---

## ⚙️ Step 5: Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings (use any text editor)
# On Windows: code .env
# On macOS/Linux: nano .env
```

### For Local Development (No Teradata):

**Your `.env` file:**
```env
# LOCAL TESTING - No Teradata needed
TERADATA_HOST=localhost
TERADATA_USER=test
TERADATA_PASSWORD=test
TERADATA_DB=local

# Local LLM (choose based on Step 4)
# For Ollama:
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=mistral

# For LM Studio:
# LLM_API_URL=http://localhost:1234/v1/chat/completions
# LLM_MODEL=local-model

# Application Settings
LOG_LEVEL=INFO
ENABLE_SQL_LOGGING=true
CACHE_ENABLED=true
KNOWLEDGE_BASE_DIR=./knowledge
```

### For Actual Teradata (With Credentials):

```env
TERADATA_HOST=10.19.199.28
TERADATA_USER=your_username
TERADATA_PASSWORD=your_password
TERADATA_DB=Tedata_temp

# Local LLM
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=mistral

LOG_LEVEL=INFO
ENABLE_SQL_LOGGING=false
CACHE_ENABLED=true
KNOWLEDGE_BASE_DIR=./knowledge
```

**⚠️ Security Note**: Never commit `.env` file to Git (it's in `.gitignore`)

---

## 🏃 Step 6: Run the Application

### Before Starting:
1. ✅ Python venv activated (see `(venv)` in terminal)
2. ✅ Dependencies installed
3. ✅ Local LLM running (Ollama, LM Studio, etc.)
4. ✅ `.env` configured

### Launch Streamlit App:

```bash
# Main application (RECOMMENDED)
streamlit run app_enhanced.py

# Or fallback CSV version (for testing without Teradata)
streamlit run app.py
```

### What You'll See:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Automatically opens in browser**, or manually visit: http://localhost:8501

---

## 💬 Step 7: Test the Agent

### If Using Sample Data (CSV - No Teradata):
1. Streamlit starts with sample data automatically
2. Go to **"Agent Chat"** tab
3. Ask: `"What's the average ARPU?"`
4. Agent processes with knowledge base + local LLM

### If Using Teradata Connection:
1. Select **Governorate** (e.g., "Giza")
2. Click **"Initialize Agent"** (loads ~30s)
3. Ask a question like:
   - `"What's our average ARPU by technology?"`
   - `"How many customers are at churn risk?"`
   - `"Break down demographics by age"`

### Expected Output:
```
📊 Visualization (chart-based on question)
Analysis: [Text response with insights]
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'src'"
**Solution**: 
```bash
# Make sure you're in project root (where app_enhanced.py is)
cd /path/to/db_bot
pwd  # Verify current directory

# Then run:
streamlit run app_enhanced.py
```

### Problem: "Connection refused" to Teradata
**Solution**: 
```bash
# Check if you have Teradata access, or skip initialization
# Use the CSV fallback: streamlit run app.py
```

### Problem: Local LLM returns errors
**Solution**:
```bash
# Check LLM is running:
# Ollama: curl http://localhost:11434
# LM Studio: curl http://localhost:1234

# If not running, start it:
# Ollama: ollama serve
# LM Studio: Click "Start Server" button
```

### Problem: "Port 8501 already in use"
**Solution**:
```bash
# Run on different port:
streamlit run app_enhanced.py --server.port 8502
```

### Problem: Slow responses
**Solution**:
```bash
# Check LLM model size:
# Use smaller model (mistral-lite, phi) instead of llama2 or neural-chat
# 
# Or disable caching in .env:
# CACHE_ENABLED=false
```

### Problem: Out of memory
**Solution**:
```bash
# Use smaller LLM model:
# Ollama: ollama pull phi  # Only 2GB
#
# Close other applications
# Increase swap (Linux):
# sudo fallocate -l 4G /swapfile
```

---

## 📁 Project Structure (Local)

```
db_bot/
├── app_enhanced.py          ← Run this (main app)
├── app.py                   ← Alternative (CSV demo)
├── requirements.txt         ← Dependencies
├── .env                     ← Your config (create from .env.example)
├── src/
│   ├── agent_tools.py       ← Analysis tools (ARPU, Churn, Migration, Demographics)
│   ├── connectors.py        ← Teradata & DuckDB connections
│   ├── knowledge_manager.py ← Knowledge base loader
│   └── visualizations.py    ← Chart generation
├── knowledge/               ← Knowledge base files
│   ├── tables/
│   ├── business/
│   ├── queries/
│   └── learnings/
└── venv/                    ← Virtual environment (created by you)
```

---

## 🔄 Development Workflow

### VS Code Setup (Recommended)

1. **Open Project**:
   ```bash
   code db_bot
   ```

2. **Select Python Interpreter**:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type: `Python: Select Interpreter`
   - Choose `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows)

3. **Install VS Code Extensions**:
   - Python (Microsoft)
   - Pylance (Microsoft)
   - Streamlit (Streamlit)
   - REST Client (optional, for API testing)

4. **Create Launch Configuration** (`.vscode/launch.json`):
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Streamlit",
         "type": "python",
         "module": "streamlit",
         "args": ["run", "app_enhanced.py"],
         "console": "integratedTerminal"
       }
     ]
   }
   ```

### Workflow Commands:
```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\Activate.ps1  # Windows

# Start Streamlit (auto-reloads on file changes)
streamlit run app_enhanced.py --logger.level=debug

# Run tests (when available)
pytest tests/ -v

# Format code
black src/ app_enhanced.py

# Check for issues
pylint src/agent_tools.py
```

---

## 🔌 Testing Without Teradata

The system works offline with sample data:

```bash
# Just run this:
streamlit run app.py

# Uses local CSV data, no database needed
# Perfect for testing UI and agent logic
```

---

## 📊 Performance Tips

### For Faster Responses:
1. **Use smaller LLM model**: `ollama pull phi` (2GB, fast)
2. **Enable caching**: `CACHE_ENABLED=true` in `.env`
3. **Use SSD** (faster than HDD for DuckDB)

### For Better Quality:
1. **Use larger LLM model**: `ollama pull neural-chat` (7GB)
2. **Increase context**: Modify knowledge base in `knowledge/business/`

---

## 🚀 Moving to Production

Once working locally:

1. **Test with Real Data**:
   ```bash
   # Update .env with actual Teradata credentials
   TERADATA_HOST=10.19.199.28
   TERADATA_USER=your_username
   TERADATA_PASSWORD=your_password
   ```

2. **Deploy to Server** (Streamlit Cloud, Docker, or VM):
   ```bash
   # Option 1: Streamlit Cloud (easiest)
   # https://streamlit.io/cloud
   
   # Option 2: Docker Container
   # See deployment section in README.md
   ```

---

## 📚 Next Steps

1. **Read**: [QUICKSTART.md](QUICKSTART.md) (5 min overview)
2. **Explore**: [README.md](README.md) (full documentation)
3. **Understand**: [VISUALIZATIONS_GUIDE.md](VISUALIZATIONS_GUIDE.md) (chart system)
4. **Deep Dive**: [AGNO_GUIDE.md](AGNO_GUIDE.md) (agent architecture)

---

## ❓ Quick Reference

| Scenario | Command |
|----------|---------|
| Start Ollama LLM | `ollama serve` |
| Run app locally | `streamlit run app_enhanced.py` |
| Test offline | `streamlit run app.py` |
| Check Python version | `python --version` |
| Activate venv (Mac/Linux) | `source venv/bin/activate` |
| Activate venv (Windows) | `.\venv\Scripts\Activate.ps1` |
| Install dependencies | `pip install -r requirements.txt` |
| Pull LLM model | `ollama pull mistral` |
| List available models | `ollama list` |

---

## 🆘 Still Stuck?

1. **Check logs**: Run with debug logging: `streamlit run app_enhanced.py --logger.level=debug`
2. **Read docs**: Start with [QUICKSTART.md](QUICKSTART.md)
3. **File structure**: Ensure `.env` is in project root, not in subdirectories
4. **LLM running**: Always verify local LLM is running before launching app

---

## 💡 Tips

- **Keep LLM running in background** while developing (separate terminal window)
- **Use `.env.example` as reference** - never edit directly
- **Test with small queries first** before complex analysis
- **Check `/knowledge` files** - they're the system's memory
- **Streamlit auto-reloads** on Python file changes (great for development)

---

*Last Updated: February 10, 2026*
*For issues or questions, refer to individual guide documents in repository root*
