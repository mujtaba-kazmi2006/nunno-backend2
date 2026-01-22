# 🚀 Deploy Nunno Finance to Vercel (Free & No Card)

Complete guide to deploy both parts of your app to **Vercel**.

---

## 🎯 Part 1: Deploy Backend to Vercel

1. **Go to [vercel.com/new](https://vercel.com/new)**
2. **Import** the `nunno-backend` repository
3. **Configure Project**:
   - **Framework Preset**: Other (or default)
   - **Environment Variables** (Expand section):
     - `OPENROUTER_API_KEY`: (Your actual API key)
     - `AI_MODEL`: `openai/gpt-4o-mini`
4. Click **Deploy**

**Note**: Since we added `vercel.json`, Vercel knows how to run it as a Python API.

✅ **Copy the URL** Vercel gives you (e.g., `https://nunno-backend.vercel.app`)

---

## 🎨 Part 2: Deploy Frontend to Vercel

### Step 1: Update Environment Variable

We need to tell the frontend where the backend lives.

1. Open `frontend/.env.production` in your editor.
2. Paste your **Backend URL** from Part 1:
   ```
   VITE_API_URL=https://nunno-backend.vercel.app
   ```
   *(Make sure there is no trailing slash `/` at the end)*

### Step 2: Push Configuration to GitHub

I (the AI) will run this command for you after you update the file, or you can run:
```bash
cd frontend
git add .
git commit -m "Update API URL"
git push
```

### Step 3: Deploy Frontend

1. **Go to [vercel.com/new](https://vercel.com/new)** again
2. **Import** the `nunno-frontend` repository
3. **Configure Project**:
   - **Framework Preset**: Vite (detected automatically)
   - **Environment Variables**:
     - `VITE_API_URL`: `https://nunno-backend.vercel.app` (Same as above)
4. Click **Deploy**

---

## ✅ Verify It Works

1. Open your new Frontend URL
2. Ask "Analyze Bitcoin"
3. If it works, you're done! 🎉

---
