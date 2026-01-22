# Scalability & Multi-User Roadmap

## 🚨 Current Status Assessment
**Can this app scale?** No.
**Can it handle multiple users?** No.

### Critical Issues Identified
1.  **Broken Deployment Dependency**: The backend relies on `betterpredictormodule.py` which lives *outside* the repository. This will fail on any cloud deployment (Vercel, Render, AWS).
2.  **No User Identity**: The system treats every request anonymously. There is no concept of "logging in" or persistent user history across devices.
3.  **Local State Risks**: Reliance on local JSON files (`predictions_db.json`) for data persistence creates race conditions (two users writing at once corrupts the file) and data loss in serverless environments (Vercel resets file systems on every deploy/restart).
4.  **Resource Bottlenecks**: Heavy technical analysis runs synchronously on the main thread or relies on global state, limiting the number of concurrent users.

---

## 🗺️ Phase 1: Architecture Cleanup (Immediate Priority)
*Goal: Make the app self-contained and deployable.*

### 1.1 Internalize Dependencies
Move the external logic into the repo so Vercel can find it.
-   **Action**: Copy `betterpredictormodule.py` and `social_scraper_module.py` (if used) into `backend/services/core_analysis/`.
-   **Refactor**: Update `technical_analysis.py` to import from this new location locally, removing `sys.path.append` hacks.

### 1.2 Eliminate Local Files
Remove reliance on `predictions_db.json` for anything other than local testing.
-   **Action**: Create a `PredictionRepository` interface.
-   **Implementation**: For now, make it in-memory only (runs fine for demo) or connect to a real DB (Phase 2).

---

## 🛠️ Phase 2: Database Implementation
*Goal: Persist data safely for multiple users.*

### 2.1 Choose a Cloud Database
Since you are using Vercel, a serverless Postgres database is recommended.
-   **Recommendation**: **Supabase** or **Neon** (Free tiers available).
-   **Why**: Fully managed, scales automatically, works great with Python (SQLAlchemy/Prisma).

### 2.2 Design Schema
Create a proper SQL schema to organized data.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR, -- 'user' or 'assistant'
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE predictions (
    id UUID PRIMARY KEY,
    ticker VARCHAR,
    result JSONB, -- Store the full analysis result
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔐 Phase 3: Authentication & Multi-User Support
*Goal: Secure user data and personalize experience.*

### 3.1 Frontend Authentication
Add a managed authentication provider. Do not rolling your own auth.
-   **Recommendation**: **Clerk** (Easiest to implement with React/Next.js).
-   **Changes**:
    -   Install `@clerk/clerk-react`.
    -   Wrap `App.tsx` in `<ClerkProvider>`.
    -   Add `<SignIn />`, `<SignUp />`, and `<UserButton />` components.

### 3.2 Backend Authorization
Secure the API endpoints.
-   **Action**: Add a middleware that verifies the JWT token sent by the frontend.
-   **Logic**:
    -   Extract `Authorization: Bearer <token>` header.
    -   Verify token with Clerk/Auth0.
    -   Get `user_id` from token.
    -   Pass `user_id` to services.

### 3.3 Context-Aware Services
Update `ChatService` to use the database.
-   **Old**: `conversation_history` passed from frontend payload.
-   **New**: `conversation_id` passed from frontend. Backend fetches history from DB (`SELECT * FROM messages WHERE conversation_id = ...`). This reduces payload size and secures history.

---

## 🚀 Phase 4: Performance & Caching
*Goal: Handle 1000+ concurrent users.*

### 4.1 Caching Layer (Redis)
Market data doesn't change every millisecond.
-   **Problem**: 100 users ask for "BTC Price" -> 100 API calls to Binance -> Rate Limited/Slow.
-   **Solution**: Cache the result of `analyze("BTCUSDT")` for 1 minute in Redis (Upstash is great for serverless).
-   **Impact**: 100 users -> 1 API call + 99 fast cache hits.

### 4.2 Async Background Tasks
If analysis becomes heavy (e.g., deep learning model), move it off the web server.
-   **Tool**: Celery or simple background tasks (`FastAPI.BackgroundTasks`).
-   **Flow**: User requests analysis -> API returns "Processing..." ID -> Frontend polls for status -> Worker completes -> API returns result.

---

## 📝 Implementation Checklist

- [ ] **Fix Imports**: Move `betterpredictormodule.py` into `backend/`.
- [ ] **Setup DB**: Create Supabase project & get connection string.
- [ ] **Backend DB Config**: Install `sqlalchemy` and connect to Supabase.
- [ ] **Setup Auth**: Create Clerk project.
- [ ] **Frontend Auth**: Integrate Clerk into React.
- [ ] **Backend Auth**: Create `get_current_user` dependency.
- [ ] **Update Chat Logic**: Read/Write messages to DB instead of memory.
