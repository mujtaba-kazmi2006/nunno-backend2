import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, Settings, PlayCircle, StopCircle, Code, RefreshCw, Menu, X } from 'lucide-react';



const CryptoChartApp = () => {
  const [chartData, setChartData] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [interval, setInterval] = useState('1m');
  const [indicators, setIndicators] = useState([]);
  const [customScript, setCustomScript] = useState(`// Example: Simple Moving Average
function calculate(data, period = 20) {
  return data.map((candle, i) => {
    if (i < period - 1) return null;
    const sum = data.slice(i - period + 1, i + 1)
      .reduce((acc, c) => acc + c.close, 0);
    return sum / period;
  });
}`);
  const [showScriptEditor, setShowScriptEditor] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [marketOverviewOpen, setMarketOverviewOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const intervalRef = useRef(null);
  const dataBufferRef = useRef([]);

  const fetchKlines = async () => {
    try {
      const response = await fetch(
        `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=100`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      const candles = data.map(k => ({
        time: new Date(k[0]).toLocaleTimeString(),
        timestamp: k[0],
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
        volume: parseFloat(k[5])
      }));

      setChartData(candles);
      dataBufferRef.current = candles;
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const startStreaming = async () => {
    setIsStreaming(true);
    await fetchKlines();
    
    // Poll every 2 seconds for updates
    intervalRef.current = setInterval(fetchKlines, 2000);
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const applyCustomIndicator = () => {
    try {
      const indicatorFunc = new Function('data', 'period', customScript + '\nreturn calculate(data, period);');
      const values = indicatorFunc(dataBufferRef.current, 20);
      
      const newIndicator = {
        id: Date.now(),
        name: 'Custom Indicator',
        color: `#${Math.floor(Math.random()*16777215).toString(16)}`,
        values
      };
      
      setIndicators(prev => [...prev, newIndicator]);
      setShowScriptEditor(false);
    } catch (error) {
      console.error('Error in custom script:', error);
      alert('Error in custom script: ' + error.message);
    }
  };

  // Recalculate indicators when data changes
  useEffect(() => {
    if (chartData.length > 0 && indicators.length > 0) {
      const updatedIndicators = indicators.map(ind => {
        try {
          const indicatorFunc = new Function('data', 'period', customScript + '\nreturn calculate(data, period);');
          const values = indicatorFunc(dataBufferRef.current, 20);
          return { ...ind, values };
        } catch {
          return ind;
        }
      });
      setIndicators(updatedIndicators);
    }
  }, [chartData.length]);

  const chartDataWithIndicators = chartData.map((candle, i) => {
    const point = { ...candle };
    indicators.forEach(ind => {
      if (ind.values[i] !== null && ind.values[i] !== undefined) {
        point[`indicator_${ind.id}`] = ind.values[i];
      }
    });
    return point;
  });

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  
  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth <= 640);
    };
    
    checkIsMobile();
    
    const handleResize = () => {
      checkIsMobile();
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4 sm:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Mobile Header */}
        <div className="sm:hidden flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-7 h-7 text-purple-400" />
            <h1 className="text-xl font-bold text-white">Nunno Finance</h1>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-400' : 'bg-gray-400'} ${isStreaming ? 'animate-pulse' : ''}`} />
            <span className="text-white text-xs">{isStreaming ? 'Live' : 'Stopped'}</span>
            <button 
              onClick={() => setMarketOverviewOpen(!marketOverviewOpen)}
              className="p-2 rounded-lg bg-slate-800 text-white"
            >
              <TrendingUp className="w-5 h-5" />
            </button>
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg bg-slate-800 text-white"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
        
        {/* Desktop Header */}
        <div className="hidden sm:flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-purple-400" />
            <h1 className="text-3xl font-bold text-white">Nunno Finance</h1>
          </div>
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${isStreaming ? 'bg-green-400' : 'bg-gray-400'} ${isStreaming ? 'animate-pulse' : ''}`} />
            <span className="text-white text-sm">{isStreaming ? 'Live' : 'Stopped'}</span>
            {lastUpdate && (
              <span className="text-purple-300 text-xs">Updated: {lastUpdate}</span>
            )}
          </div>
        </div>

        {/* Mobile Controls Menu */}
        {mobileMenuOpen && (
          <div className="sm:hidden bg-slate-800/90 backdrop-blur-sm rounded-lg p-4 mb-4 border border-purple-500/20 shadow-xl z-50 fixed inset-y-0 left-0 w-80 overflow-y-auto">
            <div className="mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold text-white">Controls</h2>
                <button 
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-1 rounded text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-purple-300 mb-1">Symbol</label>
                  <input
                    type="text"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    className="w-full px-3 py-2 bg-slate-700 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:outline-none text-sm"
                    placeholder="BTCUSDT"
                    disabled={isStreaming}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-purple-300 mb-1">Interval</label>
                  <select
                    value={interval}
                    onChange={(e) => setInterval(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-700 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:outline-none text-sm"
                    disabled={isStreaming}
                  >
                    <option value="1m">1 Min</option>
                    <option value="3m">3 Min</option>
                    <option value="5m">5 Min</option>
                    <option value="15m">15 Min</option>
                    <option value="30m">30 Min</option>
                    <option value="1h">1 Hour</option>
                    <option value="4h">4 Hours</option>
                    <option value="1d">1 Day</option>
                  </select>
                </div>
                <div className="flex gap-2">
                  {!isStreaming ? (
                    <button
                      onClick={startStreaming}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 transition-all flex items-center justify-center gap-1 text-sm"
                    >
                      <PlayCircle className="w-4 h-4" />
                      Start
                    </button>
                  ) : (
                    <button
                      onClick={stopStreaming}
                      className="flex-1 px-4 py-2 bg-gradient-to-r from-red-500 to-rose-600 text-white rounded-lg font-semibold hover:from-red-600 hover:to-rose-700 transition-all flex items-center justify-center gap-1 text-sm"
                    >
                      <StopCircle className="w-4 h-4" />
                      Stop
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setShowScriptEditor(!showScriptEditor);
                      setMobileMenuOpen(false);
                    }}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-pink-700 transition-all flex items-center justify-center gap-1 text-sm"
                  >
                    <Code className="w-4 h-4" />
                    Indicators
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Market Overview Sidebar - Right Side */}
        {marketOverviewOpen && (
          <div className="sm:hidden fixed inset-y-0 right-0 w-80 bg-slate-800/90 backdrop-blur-sm border-l border-purple-500/20 z-50 p-4 overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-white">Market Overview</h2>
              <button 
                onClick={() => setMarketOverviewOpen(false)}
                className="p-1 rounded text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-purple-300 text-xs">Symbol</p>
                <p className="text-white font-bold text-sm">{symbol}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-purple-300 text-xs">Interval</p>
                <p className="text-white font-bold text-sm">{interval}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-purple-300 text-xs">Status</p>
                <p className="text-white font-bold text-sm">{isStreaming ? 'Live' : 'Stopped'}</p>
              </div>
              
              {chartData.length > 0 && (
                <>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-purple-300 text-xs">Current Price</p>
                    <p className="text-white font-bold text-sm">${chartData[chartData.length - 1]?.close.toLocaleString()}</p>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-purple-300 text-xs">24h Change</p>
                    <p className="text-green-400 font-bold text-sm">+{(chartData[chartData.length - 1]?.close - chartData[0]?.open).toFixed(2)} ({(((chartData[chartData.length - 1]?.close - chartData[0]?.open) / chartData[0]?.open) * 100).toFixed(2)}%)</p>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-purple-300 text-xs">High (24h)</p>
                    <p className="text-green-400 font-bold text-sm">${Math.max(...chartData.map(d => d.high)).toLocaleString()}</p>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-purple-300 text-xs">Low (24h)</p>
                    <p className="text-red-400 font-bold text-sm">${Math.min(...chartData.map(d => d.low)).toLocaleString()}</p>
                  </div>
                </>
              )}
              
              {lastUpdate && (
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-purple-300 text-xs">Last Updated</p>
                  <p className="text-purple-400 font-bold text-sm">{lastUpdate}</p>
                </div>
              )}
              
              <div className="bg-slate-700/50 rounded-lg p-3">
                <p className="text-purple-300 text-xs">Active Indicators</p>
                <p className="text-white font-bold text-sm">{indicators.length}</p>
              </div>
            </div>
          </div>
        )}

        {/* Desktop Controls */}
        <div className="hidden sm:block bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 md:p-6 mb-6 border border-purple-500/20">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
            <div>
              <label className="block text-sm font-medium text-purple-300 mb-2">Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full px-4 py-2 bg-slate-700 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:outline-none"
                placeholder="BTCUSDT"
                disabled={isStreaming}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-purple-300 mb-2">Interval</label>
              <select
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
                className="w-full px-4 py-2 bg-slate-700 text-white rounded-lg border border-purple-500/30 focus:border-purple-500 focus:outline-none"
                disabled={isStreaming}
              >
                <option value="1m">1 Minute</option>
                <option value="3m">3 Minutes</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="30m">30 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
              </select>
            </div>
            <div className="flex items-end">
              {!isStreaming ? (
                <button
                  onClick={startStreaming}
                  className="w-full px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 transition-all flex items-center justify-center gap-2"
                >
                  <PlayCircle className="w-5 h-5" />
                  Start Live
                </button>
              ) : (
                <button
                  onClick={stopStreaming}
                  className="w-full px-6 py-2 bg-gradient-to-r from-red-500 to-rose-600 text-white rounded-lg font-semibold hover:from-red-600 hover:to-rose-700 transition-all flex items-center justify-center gap-2"
                >
                  <StopCircle className="w-5 h-5" />
                  Stop
                </button>
              )}
            </div>
            <div className="flex items-end">
              <button
                onClick={() => setShowScriptEditor(!showScriptEditor)}
                className="w-full px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-pink-700 transition-all flex items-center justify-center gap-2"
              >
                <Code className="w-5 h-5" />
                Indicators
              </button>
            </div>
          </div>
        </div>

        {/* Script Editor */}
        {showScriptEditor && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 sm:p-6 mb-6 border border-purple-500/20">
            <h3 className="text-xl font-bold text-white mb-4">Custom Indicator Script</h3>
            <p className="text-purple-300 text-sm mb-3">
              Write a function that takes 'data' (array of candles) and 'period' and returns an array of indicator values.
            </p>
            <textarea
              value={customScript}
              onChange={(e) => setCustomScript(e.target.value)}
              className="w-full h-40 sm:h-48 px-3 py-2 sm:px-4 sm:py-3 bg-slate-900 text-green-400 font-mono text-xs sm:text-sm rounded-lg border border-purple-500/30 focus:border-purple-500 focus:outline-none w-full"
              placeholder="Enter your indicator calculation function..."
            />
            <div className="mt-4 flex flex-col sm:flex-row gap-2 sm:gap-3">
              <button
                onClick={applyCustomIndicator}
                disabled={chartData.length === 0}
                className="px-4 py-2 sm:px-6 sm:py-2 bg-gradient-to-r from-blue-500 to-cyan-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-cyan-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                Apply Indicator
              </button>
              <button
                onClick={() => setIndicators([])}
                className="px-4 py-2 sm:px-6 sm:py-2 bg-slate-700 text-white rounded-lg font-semibold hover:bg-slate-600 transition-all text-sm"
              >
                Clear All
              </button>
            </div>
          </div>
        )}

        {/* Active Indicators */}
        {indicators.length > 0 && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 mb-6 border border-purple-500/20">
            <div className="flex items-center gap-4 flex-wrap">
              <span className="text-purple-300 font-semibold">Active:</span>
              {indicators.map(ind => (
                <div key={ind.id} className="flex items-center gap-2 bg-slate-700 px-3 py-1 rounded-full">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ind.color }} />
                  <span className="text-white text-sm">{ind.name}</span>
                  <button
                    onClick={() => setIndicators(prev => prev.filter(i => i.id !== ind.id))}
                    className="text-red-400 hover:text-red-300 text-xs ml-1 font-bold"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chart */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-purple-500/20">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
            <h2 className="text-xl sm:text-2xl font-bold text-white">
              {symbol} - {interval}
            </h2>
            {!isStreaming && chartData.length > 0 && (
              <button
                onClick={fetchKlines}
                className="px-3 py-2 sm:px-4 sm:py-2 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-all flex items-center gap-1 sm:gap-2 text-sm"
              >
                <RefreshCw className="w-3 h-3 sm:w-4 sm:h-4" />
                <span className="hidden sm:inline">Refresh</span>
                <span className="sm:hidden">R</span>
              </button>
            )}
          </div>
          
          {chartData.length === 0 ? (
            <div className="h-64 sm:h-96 flex items-center justify-center">
              <div className="text-center">
                <p className="text-purple-300 text-base sm:text-lg mb-3 sm:mb-4">Click "Start Live" to load chart data</p>
                <p className="text-purple-400 text-xs sm:text-sm">Data is fetched from Binance public API</p>
              </div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={isMobile ? 300 : 500}>
              <LineChart data={chartDataWithIndicators}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9CA3AF"
                  style={{ fontSize: isMobile ? '10px' : '12px' }}
                  interval="preserveStartEnd"
                />
                <YAxis 
                  domain={['auto', 'auto']}
                  stroke="#9CA3AF"
                  style={{ fontSize: isMobile ? '10px' : '12px' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: '1px solid #7c3aed',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '12px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="close" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  dot={false}
                  name="Price"
                  isAnimationActive={false}
                />
                {indicators.map(ind => (
                  <Line
                    key={ind.id}
                    type="monotone"
                    dataKey={`indicator_${ind.id}`}
                    stroke={ind.color}
                    strokeWidth={2}
                    dot={false}
                    name={ind.name}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}

          {/* Stats */}
          {chartData.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 sm:gap-3 mt-4 sm:mt-6">
              <div className="bg-slate-700/50 rounded-lg p-2 sm:p-3">
                <p className="text-purple-300 text-xs sm:text-sm">Open</p>
                <p className="text-white font-bold text-sm sm:text-base truncate">${chartData[chartData.length - 1]?.open.toLocaleString()}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-2 sm:p-3">
                <p className="text-purple-300 text-xs sm:text-sm">High</p>
                <p className="text-green-400 font-bold text-sm sm:text-base truncate">${chartData[chartData.length - 1]?.high.toLocaleString()}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-2 sm:p-3">
                <p className="text-purple-300 text-xs sm:text-sm">Low</p>
                <p className="text-red-400 font-bold text-sm sm:text-base truncate">${chartData[chartData.length - 1]?.low.toLocaleString()}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-2 sm:p-3">
                <p className="text-purple-300 text-xs sm:text-sm">Close</p>
                <p className="text-white font-bold text-sm sm:text-base truncate">${chartData[chartData.length - 1]?.close.toLocaleString()}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-2 sm:p-3">
                <p className="text-purple-300 text-xs sm:text-sm">Volume</p>
                <p className="text-white font-bold text-sm sm:text-base truncate">{chartData[chartData.length - 1]?.volume.toFixed(2)}</p>
              </div>
            </div>
          )}
        </div>
        
        {/* Mobile Stats Toggle - Now opens Market Overview */}
        <div className="sm:hidden mt-4 flex justify-center">
          <button 
            onClick={() => setMarketOverviewOpen(true)}
            className="px-4 py-2 bg-slate-700 text-white rounded-lg text-sm"
          >
            View Market Overview
          </button>
        </div>
      </div>
    </div>
  );
};

export default CryptoChartApp;