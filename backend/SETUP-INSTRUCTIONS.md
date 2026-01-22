# Nunno Finance - One-Click Setup & Run

## 🚀 Super Easy Setup

Just double-click this file:
```
setup-and-run.bat
```

That's it! The script will:
1. ✅ Check if Python and Node.js are installed
2. ✅ Create .env file if needed
3. ✅ Install all backend dependencies (pip)
4. ✅ Install all frontend dependencies (npm)
5. ✅ Start both servers
6. ✅ Open your browser automatically

## 📋 What You Need First

Before running the script, make sure you have:

1. **Python 3.8+** - Download from https://www.python.org/
2. **Node.js 16+** - Download from https://nodejs.org/
3. **OpenRouter API Key** - Get from https://openrouter.ai/

## 🎯 First Time Setup

1. **Double-click** `setup-and-run.bat`
2. The script will pause and ask you to add your API key
3. Open `backend\.env` in Notepad
4. Replace `your_openrouter_api_key_here` with your actual key
5. Press ENTER in the script window
6. Wait for installation (2-5 minutes)
7. Browser opens automatically!

## 🔄 Running Again Later

Just double-click `setup-and-run.bat` again!

It will skip the installation steps if everything is already installed.

## 🛑 Stopping the Servers

Just close the `setup-and-run.bat` window, or press Ctrl+C

## ⚡ Quick Start (Already Set Up)

If you've already run the setup, you can use the faster script:
```
run.bat
```

This skips the installation checks and starts immediately.

## 🆘 Troubleshooting

**"Python is not installed"**
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

**"Node.js is not installed"**
- Install Node.js from https://nodejs.org/
- Restart your computer after installation

**"Failed to install dependencies"**
- Run as Administrator
- Check your internet connection
- Try deleting `frontend\node_modules` and run again

**Servers won't start**
- Make sure ports 8000 and 5173 aren't being used
- Check firewall settings
- Look at the error messages in the server windows

## 📁 Files

- `setup-and-run.bat` - Full setup + run (use this first time)
- `run.bat` - Quick run (use after setup)
- `backend\.env` - Your API key goes here

## 🎉 That's It!

You're ready to learn finance with Nunno! 🧠
