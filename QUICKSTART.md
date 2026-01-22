# Nunno Finance - Quick Start Guide

## Prerequisites Checklist
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed  
- [ ] OpenRouter API key obtained from https://openrouter.ai/

## Setup Steps

### 1. Backend Setup (5 minutes)
```bash
cd backend
copy .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
pip install -r requirements.txt
```

### 2. Frontend Setup (3 minutes)
```bash
cd frontend
npm install
```

### 3. Run the Application
**Option A: Use the run script (Windows)**
```bash
run.bat
```

**Option B: Manual start**
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 4. Access the App
Open your browser to: **http://localhost:5173**

## First Time Usage

1. **Set your name** - Click on "Hello, User!" in the header
2. **Try a suggestion** - Click one of the suggestion chips
3. **Ask a question** - Type anything about finance or crypto
4. **Learn terms** - Click any underlined term to see definitions

## Example Questions to Try

- "Is Bitcoin a good buy right now?"
- "Explain RSI to me like I'm 15"
- "What is Ethereum's tokenomics?"
- "Help me build a $1000 crypto portfolio"
- "What's happening in the crypto market?"

## Troubleshooting

**Backend won't start?**
- Make sure you added OPENROUTER_API_KEY to .env
- Check Python version: `python --version`

**Frontend won't start?**
- Delete node_modules and run `npm install` again
- Check Node version: `node --version`

**Can't connect to backend?**
- Make sure backend is running on port 8000
- Check firewall settings

## Need Help?
Check the full README.md for detailed documentation!
