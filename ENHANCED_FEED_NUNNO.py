# Enhanced Feed Nunno Endpoint - Copy this into main.py to replace the existing feed_nunno function

@app.post("/api/v1/analyze/feed-nunno")
async def feed_nunno(request: FeedNunnoRequest):
    """
    Feed Nunno - Comprehensive Market Intelligence Report
    
    This endpoint:
    1. Fetches macro news (FED, regulations, major crypto events)
    2. Gets multi-timeframe technical analysis (15m, 1h, 4h, 1d)
    3. Calculates market temperature and volatility
    4. Streams a detailed, fact-based market intelligence report
    """
    if not chat_service or not technical_service or not news_service:
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    try:
        # Phase 1: Aggregate Data
        async def generate_feed_report():
            # Send status updates
            yield f"data: {json.dumps({'type': 'status', 'message': 'Gathering Macro News...'})}\\n\\n"
            await asyncio.sleep(0.1)
            
            # Fetch crypto-specific news
            news_data = news_service.get_news_sentiment(request.symbol)
            
            # Fetch macro economic news (FED, regulations, etc.)
            macro_headlines = []
            try:
                # Search for macro news affecting crypto
                from duckduckgo_search import DDGS
                macro_queries = [
                    "Federal Reserve interest rates crypto",
                    "cryptocurrency regulation news",
                    "Bitcoin ETF news today"
                ]
                
                with DDGS() as ddgs:
                    for query in macro_queries[:2]:  # Limit to avoid slowdown
                        try:
                            results = list(ddgs.text(query, max_results=2, timelimit='d'))
                            for r in results:
                                macro_headlines.append({
                                    "title": r.get("title", ""),
                                    "source": "Macro News",
                                    "url": r.get("href", "")
                                })
                        except:
                            continue
            except Exception as e:
                print(f"Macro news fetch error: {e}")
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing Multiple Timeframes...'})}\\n\\n"
            await asyncio.sleep(0.1)
            
            # Fetch technical analysis for multiple timeframes
            timeframes = ["15m", "1h", "4h", "1d"]
            technical_data_multi = {}
            
            for tf in timeframes:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Analyzing {tf} timeframe...'})}\\n\\n"
                    technical_data_multi[tf] = technical_service.analyze(request.symbol, tf)
                    await asyncio.sleep(0.05)
                except Exception as e:
                    print(f"Error analyzing {tf}: {e}")
                    technical_data_multi[tf] = None
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Calculating Market Metrics...'})}\\n\\n"
            await asyncio.sleep(0.1)
            
            # Calculate market temperature (price variance)
            df = technical_service.analyzer.fetch_binance_ohlcv_with_fallback(
                symbol=request.symbol, 
                interval="15m", 
                limit=96  # Last 24 hours of 15m candles
            )
            
            if len(df) > 0:
                current_price = float(df.iloc[-1]['Close'])
                avg_price_24h = float(df['Close'].mean())
                price_variance = ((current_price - avg_price_24h) / avg_price_24h) * 100
                volatility = float(df['Close'].std())
                high_24h = float(df['High'].max())
                low_24h = float(df['Low'].min())
                volume_24h = float(df['Volume'].sum())
            else:
                current_price = 0
                price_variance = 0
                volatility = 0
                high_24h = 0
                low_24h = 0
                volume_24h = 0
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Compiling Intelligence Report...'})}\\n\\n"
            await asyncio.sleep(0.1)
            
            # Phase 2: Build the Comprehensive AI Prompt
            coin_name = request.symbol.replace("USDT", "").replace("USD", "")
            
            # Format crypto news
            crypto_news_text = "\\n".join([
                f"- {h.get('title', 'N/A')} (Source: {h.get('source', 'Unknown')})"
                for h in news_data.get('headlines', [])[:5]
            ])
            
            # Format macro news
            macro_news_text = "\\n".join([
                f"- {h.get('title', 'N/A')}"
                for h in macro_headlines[:5]
            ])
            
            # Build multi-timeframe technical summary
            tf_summary = ""
            for tf, data in technical_data_multi.items():
                if data:
                    ind = data.get('indicators', {})
                    tf_summary += f"""
**{tf.upper()} Timeframe:**
- Bias: {data.get('bias', 'N/A')} ({data.get('confidence', 0):.1f}% confidence)
- RSI: {ind.get('rsi_14', 0):.1f}
- MACD: {ind.get('macd', 0):.2f} | Signal: {ind.get('macd_signal', 0):.2f} | Histogram: {ind.get('macd_histogram', 0):.2f}
- ADX: {ind.get('adx', 0):.1f} (Trend Strength)
- Stochastic K/D: {ind.get('stoch_k', 0):.1f} / {ind.get('stoch_d', 0):.1f}
- Volume Ratio: {ind.get('volume_ratio', 0):.2f}x
- EMA 9/21/50: ${ind.get('ema_9', 0):.2f} / ${ind.get('ema_21', 0):.2f} / ${ind.get('ema_50', 0):.2f}
- Bollinger Bands: Upper ${ind.get('bb_upper', 0):.2f} | Middle ${ind.get('bb_middle', 0):.2f} | Lower ${ind.get('bb_lower', 0):.2f}
- BB Position: {ind.get('bb_position', 0):.2f}
- ATR: ${ind.get('atr', 0):.2f} ({ind.get('atr_percent', 0):.2f}% of price)
- Support: ${data.get('support_levels', [0])[0]:.2f} | Resistance: ${data.get('resistance_levels', [0])[0]:.2f}
"""
            
            # Build structured prompt for detailed analysis
            prompt = f"""You are Nunno, a professional crypto market analyst. Generate a comprehensive MARKET INTELLIGENCE REPORT using ONLY the data provided below. Be factual, structured, and detailed.

═══════════════════════════════════════════════════════
📊 MARKET INTELLIGENCE REPORT: {request.symbol}
═══════════════════════════════════════════════════════

**CURRENT MARKET DATA:**
- Symbol: {request.symbol}
- Current Price: ${current_price:,.2f}
- 24h High: ${high_24h:,.2f}
- 24h Low: ${low_24h:,.2f}
- 24h Volume: {volume_24h:,.0f}
- Price Variance (24h): {price_variance:+.2f}%
- Volatility (Std Dev): ${volatility:,.2f}

**MARKET SENTIMENT:**
- Fear & Greed Index: {news_data.get('fear_greed_index', {}).get('value', 'N/A')}/100 ({news_data.get('fear_greed_index', {}).get('classification', 'N/A')})
- Overall Sentiment: {news_data.get('sentiment', 'Neutral')}

**MACRO NEWS & EVENTS:**
{macro_news_text if macro_news_text else "No major macro news in the last 24 hours"}

**CRYPTO-SPECIFIC NEWS:**
{crypto_news_text if crypto_news_text else "No recent crypto-specific news available"}

**MULTI-TIMEFRAME TECHNICAL ANALYSIS:**
{tf_summary}

═══════════════════════════════════════════════════════

**REPORT STRUCTURE (Use this exact format):**

## 📰 News Summary
[Summarize the key macro and crypto news. If no news, state "No significant news in the last 24 hours affecting {coin_name}"]

## 📈 Multi-Timeframe Analysis
[Create a table comparing the 4 timeframes. Format as markdown table:]
| Timeframe | Bias | RSI | MACD Signal | Trend Strength (ADX) | Key Level |
|-----------|------|-----|-------------|---------------------|-----------|
[Fill with data from each timeframe]

## 🔍 Detailed Indicator Breakdown (15m Focus)
[List ALL indicators with their values and what they mean - be educational but concise]

## 🎯 Market Temperature
[Explain the price variance, volatility, and what it means for traders]

## 💡 Key Observations
[3-5 bullet points of the most important facts from the data]

## ⚠️ Risk Factors
[Based on volatility, sentiment, and technical data]

═══════════════════════════════════════════════════════

**IMPORTANT RULES:**
1. Use ONLY the data provided above
2. Be factual - no speculation
3. Use markdown tables and formatting
4. Keep it professional and structured
5. Maximum 500 words
6. If data is missing, state "Data unavailable" instead of making assumptions

Generate the report now:"""

            # Phase 3: Stream AI Response
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating Report...'})}\\n\\n"
            await asyncio.sleep(0.1)
            
            # Use chat service to stream the AI response
            async for chunk in chat_service.stream_message(
                message=prompt,
                user_name=request.user_name,
                user_age=18,
                conversation_history=[]
            ):
                # Forward the chunk directly
                yield chunk
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\\n\\n"
        
        return StreamingResponse(
            generate_feed_report(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"Feed Nunno error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
