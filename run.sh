#!/bin/bash

echo "===================================="
echo "  Starting Nunno Finance"
echo "===================================="
echo ""

echo "[1/2] Starting Backend Server..."
cd backend
python main.py &
BACKEND_PID=$!

sleep 3

echo "[2/2] Starting Frontend Server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "===================================="
echo "  Nunno Finance is running!"
echo "===================================="
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
