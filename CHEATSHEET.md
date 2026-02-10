# ⚡ Quick Start Cheatsheet (Local Development)

## 🎯 30-Second Setup

```bash
# 1. Clone project
git clone https://github.com/chatguide26/db_bot.git
cd db_bot

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
.\venv\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your local LLM URL

# 5. Start LLM (in separate terminal)
ollama serve  # if using Ollama

# 6. Run app
streamlit run app_enhanced.py
```

---

## 🤖 Local LLM Options (Pick One)

### Ollama (Recommended)
```bash
# Install from: https://ollama.ai
ollama pull mistral    # Fast & good (2GB)
ollama serve          # Runs on http://localhost:11434
```

### LM Studio
```bash
# Download from: https://lmstudio.ai
# 1. Launch app
# 2. Select & download a model
# 3. Click "Start Server"
# Runs on: http://localhost:1234
```

---

## ⚙️ Configuration (`.env`)

```env
# For local development with Ollama
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL=mistral

# For local development with LM Studio
# LLM_API_URL=http://localhost:1234/v1/chat/completions
# LLM_MODEL=local-model
```

---

## 🚀 Running the App

```bash
# Make sure virtual environment is activated
# Make sure local LLM is running

# Option 1: Main app (recommended)
streamlit run app_enhanced.py

# Option 2: Demo with sample data (no database)
streamlit run app.py

# Option 3: Debug mode
streamlit run app_enhanced.py --logger.level=debug
```

Visit: **http://localhost:8501**

---

## 💬 Test Questions

```
"What's the average ARPU?"
"How many customers are at risk?"
"Show demographic breakdown"
"Compare technologies by revenue"
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `cd` to project root, check `pwd` |
| Connection refused | Start LLM: `ollama serve` in new terminal |
| Port 8501 in use | `streamlit run app_enhanced.py --server.port 8502` |
| Slow responses | Use smaller model: `ollama pull phi` |
| Out of memory | Close other apps, use smaller model |

---

## 📚 Full Guides

- **[LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)** - Complete setup with troubleshooting
- **[QUICKSTART.md](QUICKSTART.md)** - First queries & overview  
- **[README.md](README.md)** - Full documentation
- **[INDEX.md](INDEX.md)** - Navigate all docs

---

## 🔗 Useful Links

| Resource | URL |
|----------|-----|
| Ollama | https://ollama.ai |
| LM Studio | https://lmstudio.ai |
| Streamlit | https://streamlit.io |
| Project Repo | https://github.com/chatguide26/db_bot |

---

## ✅ Checklist Before Running

- [ ] Python 3.10+ installed
- [ ] Virtual environment created & activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created & configured
- [ ] Local LLM downloaded & running
- [ ] LLM endpoint URL correct in `.env`

---

**You're ready! Run:** `streamlit run app_enhanced.py`

*For detailed setup, see [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)*
