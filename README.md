# Nunno Finance

**Your Empathetic AI Financial Educator** 🧠

Nunno Finance is a full-stack application that makes learning about trading and investing simple and accessible for complete beginners. Powered by Claude Opus 4.5 via OpenRouter, it provides beginner-friendly explanations with real-world analogies for every technical term.

## 🌟 Features

- **💬 Chat-First Interface**: Natural conversation with an AI that explains finance like you're 15
- **📊 Technical Analysis**: Real-time crypto analysis with beginner-friendly explanations
- **💰 Tokenomics Analysis**: Deep dive into token economics made simple
- **📰 Market Sentiment**: Fear & Greed Index and news analysis
- **🌡️ Market Temperature Gauge**: Visual representation of market emotions
- **💡 Learn More Popups**: Click any financial term to see simple definitions and analogies
- **🎯 Educational Cards**: Color-coded, actionable insights with confidence levels

## 🏗️ Architecture

- **Backend**: FastAPI with Python
  - Technical analysis service (from `betterpredictormodule.py`)
  - Chat orchestration with OpenRouter
  - Tokenomics and news services
  
- **Frontend**: React + Vite
  - Modern, polished UI with smooth animations
  - Recharts for visualizations
  - Responsive design for all devices

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **OpenRouter API Key** (Get from [https://openrouter.ai/](https://openrouter.ai/))

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend

# Copy environment template
copy .env.example .env

# Edit .env and add your OPENROUTER_API_KEY

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The backend will start at `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at `http://localhost:5173`

### 3. Access the Application

Open your browser and navigate to `http://localhost:5173`

## 🔑 Environment Variables

Create a `.env` file in the `backend` directory:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional
AI_MODEL=anthropic/claude-3.5-sonnet  # Update to opus when available
NEWS_API_KEY=your_news_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key_here
```

## 📁 Project Structure

```
NunnoFinance/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── services/
│   │   ├── technical_analysis.py
│   │   ├── chat_service.py
│   │   ├── tokenomics_service.py
│   │   └── news_service.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── MarketTemperature.jsx
│   │   │   ├── EducationalCard.jsx
│   │   │   └── TermDefinition.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── styles/
│   │       └── App.css
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

## 🎯 Usage Examples

### Ask About Crypto Analysis
```
"Is Bitcoin a good buy right now?"
```
→ Nunno will analyze Bitcoin's technical indicators and explain them in simple terms

### Learn About Concepts
```
"What is RSI?"
```
→ Get a simple explanation with real-world analogies

### Build a Portfolio
```
"Help me build a crypto portfolio with $1000"
```
→ Get educational portfolio examples with risk explanations

## 🎨 Key Features Explained

### The Empathetic Expert Persona
Every response includes:
- **Beginner's Notes**: Automatic explanations for technical terms
- **Real-world Analogies**: "RSI is like a thermometer for the market"
- **Simple Language**: No jargon without explanation

### Tool Calling
The AI automatically uses the right tools:
- Asks about price → Calls technical analysis
- Asks about tokenomics → Calls tokenomics service
- Asks about market → Calls news service

### Visual Education
- **Market Temperature Gauge**: Shows Fear & Greed Index visually
- **Educational Cards**: Color-coded analysis results
- **Confidence Indicators**: Shows how certain the analysis is

## 🛠️ Development

### Backend Development
```bash
cd backend
python main.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Building for Production
```bash
cd frontend
npm run build
```

## 📝 API Documentation

Once the backend is running, visit:
- API Docs: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

### Main Endpoints

- `POST /api/v1/chat` - Chat with Nunno AI
- `GET /api/v1/technical/{ticker}` - Get technical analysis
- `GET /api/v1/tokenomics/{coin_id}` - Get tokenomics data
- `GET /api/v1/news/{ticker}` - Get news and sentiment

## 🤝 Contributing

This project was built by Mujtaba Kazmi as an educational tool for beginners.

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Claude Opus 4.5** via OpenRouter for AI capabilities
- **Recharts** for beautiful visualizations
- **FastAPI** for the robust backend
- **React** for the interactive frontend

## 💡 Tips for Best Experience

1. **Be Specific**: Ask about specific cryptocurrencies or concepts
2. **Ask Follow-ups**: Nunno remembers your conversation
3. **Click Terms**: Click any financial term to learn more
4. **Check the Gauge**: Watch the Market Temperature for sentiment

## 🐛 Troubleshooting

### Backend won't start
- Check that Python 3.8+ is installed
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Make sure `.env` file exists with valid API key

### Frontend won't start
- Check that Node.js 16+ is installed
- Delete `node_modules` and run `npm install` again
- Make sure backend is running on port 8000

### API errors
- Verify OpenRouter API key is valid
- Check internet connection
- Look at backend console for error messages

## 📧 Support

For questions or issues, please check the console logs for detailed error messages.

---

**Built with ❤️ to make finance accessible for everyone**
