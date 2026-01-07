
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Bar, Line, ReferenceLine } from 'recharts';

import { CoinData, HistoryPoint } from '../types';
import { RotateCcw, ZoomIn, MessageCircle, X, Send, Sparkles, Loader2, Hand, Bell, History, BrainCircuit, Eye, EyeOff } from 'lucide-react';
import { askChartAnalyst } from '../services/apiService';

interface MarketChartProps {
  data: CoinData['history'];
  color: string;
  type: 'area' | 'candle';
  showRSI?: boolean;
  showMACD?: boolean;
  showBB?: boolean;
  onTimeRangeChange?: (range: string) => void;
  onAddAlert?: (price: number) => void;
  onAnalyzeHistory?: (date: string) => void;
}

// Custom Candle Shape
const Candle = (props: any) => {
  const { x, y, width, height, payload } = props;
  const { open, high, low } = payload;
  const close = payload.close ?? payload.price;

  if (open === undefined || close === undefined || high === undefined || low === undefined) return null;

  const isUp = close >= open;
  // Vietnamese/Asian convention: Red = Up (bullish), Green = Down (bearish)
  const color = isUp ? '#10b981' : '#f43f5e';

  const range = high - low;
  if (range === 0) return null;

  const pixelPerUnit = height / range;

  const openOffset = (high - open) * pixelPerUnit;
  const closeOffset = (high - close) * pixelPerUnit;

  const bodyTop = y + Math.min(openOffset, closeOffset);
  const bodyLength = Math.max(2, Math.abs(openOffset - closeOffset));

  return (
    <g>
      <line x1={x + width / 2} y1={y} x2={x + width / 2} y2={y + height} stroke={color} strokeWidth={1} />
      <rect x={x} y={bodyTop} width={width} height={bodyLength} fill={color} stroke="none" rx={1} />
    </g>
  );
};

// New Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const price = data.price;
    const forecast = data.forecast;
    // Determine if this point is purely future (no real price)
    const isFuture = price === undefined && forecast !== undefined;

    return (
      <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-xl max-w-[280px] backdrop-blur-sm bg-opacity-95 z-50 pointer-events-none">
        <p className="text-slate-500 text-xs mb-1 font-medium tracking-wide">{label}</p>

        {/* Price & Forecast Display */}
        <div className="flex flex-col gap-1 mb-2">
          {/* Real Price - Only show if available */}
          {price !== undefined && (
            <div className="flex justify-between items-baseline gap-4">
              <span className="text-xs text-slate-500 font-semibold uppercase">Price</span>
              <p className="text-slate-900 font-bold text-lg">
                ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
            </div>
          )}

          {/* Volume Display */}
          {data.volume !== undefined && (
            <div className="flex justify-between items-center text-xs mt-1 pb-2 border-b border-slate-100">
              <span className="text-slate-500 font-semibold uppercase">Vol</span>
              <span className="text-slate-700 font-mono">
                {new Intl.NumberFormat('en-US', { notation: "compact", maximumFractionDigits: 1 }).format(data.volume)}
              </span>
            </div>
          )}

          {/* Technical Indicators */}
          {(data.rsi || data.macd) && (
            <div className="pt-2 border-b border-slate-100 pb-2 mb-1">
              {data.rsi && (
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className="text-purple-600 font-bold uppercase">RSI</span>
                  <span className="text-slate-600 font-mono">{data.rsi.toFixed(2)}</span>
                </div>
              )}
              {data.macd && (
                <div className="flex justify-between items-center text-xs">
                  <span className="text-orange-500 font-bold uppercase">MACD</span>
                  <span className="text-slate-600 font-mono">{data.macd.toFixed(2)}</span>
                </div>
              )}
            </div>
          )}

          {/* Forecast - Always show if available */}
          {forecast !== undefined && (
            <div className="flex justify-between items-baseline gap-4 mt-1">
              <span className="text-xs text-purple-600 font-semibold uppercase flex items-center gap-1">
                Prediction
                {isFuture && <span className="bg-purple-100 text-purple-700 text-[8px] px-1 rounded ml-1">FUTURE</span>}
              </span>
              <p className="text-purple-600 font-bold text-lg">
                ${forecast.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
            </div>
          )}
        </div>

        {/* OHLC Section - Collapsed if candle chart is relevant */}
        {data.open !== undefined && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 my-3 text-[10px] text-slate-500 border-t border-b border-slate-100 py-2">
            <div className="flex justify-between"><span>Open:</span> <span className="text-slate-800">${data.open.toFixed(2)}</span></div>
            <div className="flex justify-between"><span>High:</span> <span className="text-slate-800">${data.high?.toFixed(2)}</span></div>
            <div className="flex justify-between"><span>Low:</span> <span className="text-slate-800">${data.low?.toFixed(2)}</span></div>
            <div className="flex justify-between"><span>Close:</span> <span className="text-slate-800">${(data.price ?? 0).toFixed(2)}</span></div>
          </div>
        )}

        {/* AI Insight Section */}
        {(data.newsSummary || data.sentimentScore !== undefined) && (
          <div className="mt-2 pt-2 border-t border-slate-100">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] uppercase font-bold text-indigo-600 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-pulse"></span>
                AI Insight
              </span>
              {data.sentimentScore !== undefined && (
                <div className={`flex items-center gap-1.5 px-1.5 py-0.5 rounded ${data.sentimentScore > 50 ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
                  <span className="text-[10px] font-bold">{data.sentimentScore}/100</span>
                </div>
              )}
            </div>
            {data.newsSummary ? (
              <p className="text-xs text-slate-600 italic leading-snug">
                "{data.newsSummary}"
              </p>
            ) : (
              <p className="text-[10px] text-slate-400 italic">No major news events detected.</p>
            )}
          </div>
        )}
      </div>
    );
  }
  return null;
};

const MarketChart: React.FC<MarketChartProps> = ({
  data, color, type, showRSI, showMACD, showBB,
  onTimeRangeChange, onAddAlert, onAnalyzeHistory
}) => {
  // AI Chat State
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatAnswer, setChatAnswer] = useState<string | null>(null);
  const [isThinking, setIsThinking] = useState(false);

  // Forecast Visibility State
  const [showForecast, setShowForecast] = useState(true);

  // Zooming & Panning State
  const [startIndex, setStartIndex] = useState<number>(0);
  const [endIndex, setEndIndex] = useState<number>(0);
  const chartRef = useRef<HTMLDivElement>(null);

  // Dragging State
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartIndex, setDragStartIndex] = useState(0);
  const [dragEndIndex, setDragEndIndex] = useState(0);

  // Interaction State
  const [hoveredDataPoint, setHoveredDataPoint] = useState<any>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number, y: number, data: any } | null>(null);

  // Initialize Zoom State - only reset on significant data length changes
  const prevDataLengthRef = useRef(0);
  useEffect(() => {
    const prevLen = prevDataLengthRef.current;

    if (data.length > 0) {
      // First load or significant change (range switch, new coin)
      if (prevLen === 0 || Math.abs(data.length - prevLen) > 2) {
        setStartIndex(0);
        setEndIndex(data.length - 1);
      }
      // For real-time updates (same length or +/-1), keep current view
      // No action needed - visibleData will update automatically
    }

    prevDataLengthRef.current = data.length;
  }, [data.length]); // Only check length to avoid excessive re-runs

  // Handle Global Click to close context menu
  useEffect(() => {
    const handleClick = () => setContextMenu(null);
    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, []);

  // --- Technical Indicator Calculations ---
  const calculatedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Copy data to avoid mutation
    const points: any[] = data.map(d => ({ ...d }));

    // Helper: SMA
    const calculateSMA = (period: number, index: number, source: any[]) => {
      if (index < period - 1) return null;
      let sum = 0;
      for (let i = 0; i < period; i++) {
        const p = source[index - i].price;
        if (typeof p !== 'number') return null;
        sum += p;
      }
      return sum / period;
    };

    // 1. Bollinger Bands (20 SMA, 2 StdDev)
    if (showBB) {
      for (let i = 0; i < points.length; i++) {
        const sma = calculateSMA(20, i, points);
        if (sma !== null) {
          let sumSqDiff = 0;
          let valid = true;
          for (let j = 0; j < 20; j++) {
            const p = points[i - j].price;
            if (typeof p !== 'number') { valid = false; break; }
            sumSqDiff += Math.pow(p - sma, 2);
          }

          if (valid) {
            const stdDev = Math.sqrt(sumSqDiff / 20);
            points[i].bbUpper = sma + (2 * stdDev);
            points[i].bbLower = sma - (2 * stdDev);
            points[i].bbMiddle = sma;
          }
        }
      }
    }

    // 2. RSI (14)
    if (showRSI) {
      let gains: number[] = [];
      let losses: number[] = [];

      for (let i = 0; i < points.length; i++) {
        if (i === 0) {
          points[i].rsi = null;
          continue;
        }

        const currentPrice = points[i].price;
        const prevPrice = points[i - 1].price;

        if (typeof currentPrice !== 'number' || typeof prevPrice !== 'number') {
          points[i].rsi = null;
          // Maintain array synchronization if needed, or simply skip
          gains.push(0);
          losses.push(0);
          continue;
        }

        const change = currentPrice - prevPrice;
        gains.push(change > 0 ? change : 0);
        losses.push(change < 0 ? Math.abs(change) : 0);

        if (i >= 14) {
          const avgGain = gains.slice(i - 14, i).reduce((a, b) => a + b, 0) / 14;
          const avgLoss = losses.slice(i - 14, i).reduce((a, b) => a + b, 0) / 14;
          if (avgLoss === 0) {
            points[i].rsi = 100;
          } else {
            const rs = avgGain / avgLoss;
            points[i].rsi = 100 - (100 / (1 + rs));
          }
        }
      }
    }

    // 3. MACD (12, 26, 9)
    if (showMACD) {
      // Note: MACD logic simplified, assumes data starts with price. 
      // If undefined encountered, we break or skip.
      let ema12 = typeof points[0].price === 'number' ? points[0].price : 0;
      let ema26 = typeof points[0].price === 'number' ? points[0].price : 0;
      let signalEma = 0;
      let initialized = typeof points[0].price === 'number';

      for (let i = 0; i < points.length; i++) {
        const price = points[i].price;

        if (typeof price !== 'number') {
          continue;
        }

        if (!initialized) {
          ema12 = price;
          ema26 = price;
          initialized = true;
          continue;
        }

        ema12 = (price - ema12) * (2 / (12 + 1)) + ema12;
        ema26 = (price - ema26) * (2 / (26 + 1)) + ema26;

        const macdLine = ema12 - ema26;
        points[i].macd = macdLine;

        if (i === 0) signalEma = macdLine;
        signalEma = (macdLine - signalEma) * (2 / (9 + 1)) + signalEma;
        points[i].macdSignal = signalEma;
        points[i].macdHist = macdLine - signalEma;
      }
    }

    return points.map(d => ({
      ...d,
      candleRange: [d.low ?? d.price, d.high ?? d.price]
    }));
  }, [data, showRSI, showMACD, showBB]);

  const visibleData = calculatedData.slice(startIndex, endIndex + 1);

  // Time Range Check for Infinite Scroll or Auto-switch
  const checkTimeRange = (startIdx: number, endIdx: number) => {
    if (!onTimeRangeChange || startIdx >= endIdx || !data[startIdx] || !data[endIdx]) return;

    const startTs = data[startIdx].ts;
    const endTs = data[endIdx].ts;
    const duration = Math.abs(endTs - startTs);

    const ONE_DAY = 24 * 60 * 60 * 1000;
    const ONE_MONTH = 30 * ONE_DAY;
    const THREE_MONTHS = 90 * ONE_DAY;
    const ONE_YEAR = 365 * ONE_DAY;

    let newRange = null;
    if (duration <= ONE_DAY * 1.5) newRange = '1D';
    else if (duration <= ONE_MONTH * 1.5) newRange = '1M';
    else if (duration <= THREE_MONTHS * 1.5) newRange = '3M';
    else if (duration <= ONE_YEAR * 1.5) newRange = '1Y';
    else if (duration <= ONE_YEAR * 5.5) newRange = '5Y';

    if (newRange) onTimeRangeChange(newRange);
  };

  // Wheel Zoom
  const handleWheel = (e: React.WheelEvent) => {
    const currentPoints = endIndex - startIndex;
    const zoomSpeed = Math.max(1, Math.floor(currentPoints * 0.1));

    let newStart = startIndex;
    let newEnd = endIndex;

    if (e.deltaY < 0) {
      // Zoom In
      newStart = Math.min(startIndex + zoomSpeed, endIndex - 5);
      newEnd = Math.max(endIndex - zoomSpeed, startIndex + 5);
    } else {
      // Zoom Out
      newStart = Math.max(0, startIndex - zoomSpeed);
      newEnd = Math.min(data.length - 1, endIndex + zoomSpeed);
    }

    if (newStart !== startIndex || newEnd !== endIndex) {
      setStartIndex(newStart);
      setEndIndex(newEnd);

      // Check for range switch if hitting limits
      if ((newStart === 0 && newEnd === data.length - 1 && e.deltaY > 0) || (e.deltaY < 0 && (newEnd - newStart) < 10)) {
        checkTimeRange(newStart, newEnd);
      }
    }
  };

  // Mouse Drag Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    // If Context Menu click, do not drag
    if (e.button === 2) return;

    setIsDragging(true);
    setDragStartX(e.clientX);
    setDragStartIndex(startIndex);
    setDragEndIndex(endIndex);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !chartRef.current || e.buttons !== 1) {
      if (isDragging) setIsDragging(false);
      return;
    }

    const width = chartRef.current.clientWidth;
    const deltaX = e.clientX - dragStartX;

    const visiblePoints = dragEndIndex - dragStartIndex;
    const pointsPerPixel = visiblePoints / width;
    const shift = Math.round(deltaX * pointsPerPixel);

    const shiftDir = -shift;

    let newStart = dragStartIndex + shiftDir;
    let newEnd = dragEndIndex + shiftDir;

    // Clamping
    if (newStart < 0) {
      const diff = 0 - newStart;
      newStart = 0;
      newEnd = Math.min(data.length - 1, newEnd + diff);
    }
    if (newEnd > data.length - 1) {
      const diff = newEnd - (data.length - 1);
      newEnd = data.length - 1;
      newStart = Math.max(0, newStart - diff);
    }

    setStartIndex(newStart);
    setEndIndex(newEnd);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Helper to fallback find data point by X coordinate
  const getPointFromCoordinate = (x: number) => {
    if (!chartRef.current || visibleData.length === 0) return null;
    const rect = chartRef.current.getBoundingClientRect();
    const width = rect.width;

    // Approximate margins based on Axis config
    const leftMargin = (showRSI || showMACD) ? 40 : 0;
    const rightMargin = 60; // YAxis Price
    const chartWidth = width - leftMargin - rightMargin;

    const relativeX = x - rect.left - leftMargin;

    if (relativeX < 0 || relativeX > chartWidth) return null;

    const ratio = relativeX / chartWidth;
    const index = Math.floor(ratio * visibleData.length);

    return visibleData[Math.min(Math.max(0, index), visibleData.length - 1)];
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();

    const point = hoveredDataPoint || getPointFromCoordinate(e.clientX);

    if (point) {
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        data: point
      });
    }
  };

  // Action Handlers
  const handleMenuAlert = () => {
    if (contextMenu && onAddAlert) {
      onAddAlert(contextMenu.data.price ?? contextMenu.data.forecast);
    }
    setContextMenu(null);
  };

  const handleMenuHistory = () => {
    if (contextMenu && onAnalyzeHistory) {
      onAnalyzeHistory(contextMenu.data.time);
    }
    setContextMenu(null);
  };

  const handleResetZoom = () => {
    setStartIndex(0);
    setEndIndex(data.length - 1);
  };

  const handleAskAI = async () => {
    if (!chatInput.trim()) return;
    setIsThinking(true);
    setChatAnswer(null);
    try {
      const answer = await askChartAnalyst('Current Asset', visibleData, chatInput);
      setChatAnswer(answer);
    } catch (e) {
      setChatAnswer("Sorry, I couldn't analyze the chart right now.");
    } finally {
      setIsThinking(false);
    }
  };

  // Determine if forecast data exists in current visible data
  const hasForecast = useMemo(() => {
    return visibleData.some(d => d.forecast !== undefined);
  }, [visibleData]);

  return (
    <div
      ref={chartRef}
      className={`h-full w-full relative group overflow-hidden select-none ${isDragging ? 'cursor-grabbing' : 'cursor-default'}`}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onContextMenu={handleContextMenu}
    >

      {/* Controls Overlay */}
      <div className="absolute top-2 right-16 z-10 flex gap-2">
        {/* Forecast Toggle */}
        {hasForecast && (
          <button
            onClick={(e) => { e.stopPropagation(); setShowForecast(!showForecast); }}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md backdrop-blur-sm border shadow-sm transition-all ${showForecast ? 'bg-purple-600 border-purple-500 text-white hover:bg-purple-700' : 'bg-white/80 border-slate-200 text-slate-600 hover:text-purple-600'}`}
            title={showForecast ? "Hide Prediction" : "Show Prediction"}
          >
            {showForecast ? <BrainCircuit size={12} /> : <EyeOff size={12} />}
            <span className="hidden sm:inline">Prediction</span>
          </button>
        )}

        <button
          onClick={(e) => { e.stopPropagation(); setIsChatOpen(!isChatOpen); }}
          className={`flex items-center gap-1 bg-white/80 hover:bg-indigo-600 text-slate-700 hover:text-white text-xs px-3 py-1.5 rounded-md backdrop-blur-sm border border-slate-200 shadow-sm transition-all ${isChatOpen ? 'bg-indigo-600 border-indigo-500 text-white' : ''}`}
        >
          <Sparkles size={12} /> Ask AI
        </button>

        <button
          onClick={(e) => { e.stopPropagation(); handleResetZoom(); }}
          className="flex items-center gap-1 bg-white/80 hover:bg-slate-100 text-slate-700 text-xs px-3 py-1.5 rounded-md backdrop-blur-sm border border-slate-200 shadow-sm transition-all"
          title="Reset Zoom"
        >
          <RotateCcw size={12} />
        </button>
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="fixed z-50 bg-white border border-slate-200 rounded-lg shadow-xl w-48 py-1 overflow-hidden animate-in fade-in zoom-in-95 duration-100"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div className="px-3 py-2 border-b border-slate-100 bg-slate-50">
            <span className="text-[10px] font-bold text-slate-500 uppercase block">
              {contextMenu.data.time.split(',')[0]}
            </span>
            <span className="text-sm font-bold text-slate-800">
              ${(contextMenu.data.price ?? contextMenu.data.forecast ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <button
            onClick={handleMenuAlert}
            className="w-full flex items-center gap-2 px-3 py-2.5 text-xs text-slate-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
          >
            <Bell size={14} /> Add Price Alert
          </button>
          <button
            onClick={handleMenuHistory}
            className="w-full flex items-center gap-2 px-3 py-2.5 text-xs text-slate-700 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
          >
            <History size={14} /> What happened this time?
          </button>
        </div>
      )}

      {/* Embedded Chat Overlay */}
      {isChatOpen && (
        <div className="absolute bottom-12 right-12 z-20 w-80 bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl shadow-xl p-4 animate-in fade-in slide-in-from-bottom-4" onMouseDown={e => e.stopPropagation()}>
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold text-indigo-600 flex items-center gap-1">
              <Sparkles size={12} /> Chart Intelligence
            </span>
            <button onClick={() => setIsChatOpen(false)} className="text-slate-400 hover:text-slate-600">
              <X size={14} />
            </button>
          </div>

          {chatAnswer && (
            <div className="bg-indigo-50 border border-indigo-100 p-3 rounded-lg mb-3">
              <p className="text-sm text-slate-700 leading-relaxed">{chatAnswer}</p>
            </div>
          )}

          <div className="relative">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAskAI()}
              placeholder="e.g. Is this a bullish trend?"
              className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 pl-3 pr-9 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
            />
            <button
              onClick={handleAskAI}
              disabled={isThinking || !chatInput.trim()}
              className="absolute right-1.5 top-1.5 p-1 bg-indigo-600 rounded text-white disabled:opacity-50 hover:bg-indigo-700"
            >
              {isThinking ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
            </button>
          </div>
        </div>
      )}

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          key={`chart-${visibleData.length > 0 ? visibleData[visibleData.length - 1]?.price || Date.now() : Date.now()}`}
          data={visibleData}
          margin={{ top: 10, right: 0, left: 0, bottom: 0 }}
          onMouseMove={(e: any) => {
            if (e.activePayload && e.activePayload.length > 0) {
              setHoveredDataPoint(e.activePayload[0].payload);
            }
          }}
        >
          <defs>
            <linearGradient id={`color${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorRSI" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis
            dataKey="time"
            stroke="#94a3b8"
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
            minTickGap={30}
            tickFormatter={(val) => {
              if (typeof val === 'string' && val.split(':').length === 3) {
                const parts = val.split(':');
                return `${parts[0]}:${parts[1]}`;
              }
              return val;
            }}
          />

          {/* Main Price Axis (Right) */}
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={['auto', 'auto']}
            stroke="#94a3b8"
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `$${value}`}
            width={60}
          />

          {/* Secondary Axis for Indicators (Left) */}
          {(showRSI || showMACD) && (
            <YAxis
              yAxisId="left"
              orientation="left"
              domain={showRSI ? [0, 100] : ['auto', 'auto']}
              stroke="#64748b"
              tick={{ fontSize: 10, fill: '#64748b' }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
          )}

          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: '#94a3b8', strokeWidth: 1, strokeDasharray: '4 4' }}
            wrapperStyle={{ outline: 'none' }}
          />

          {/* Horizontal Crosshair Line */}
          {hoveredDataPoint && (hoveredDataPoint.price !== undefined || hoveredDataPoint.forecast !== undefined) && (
            <ReferenceLine
              yAxisId="right"
              y={hoveredDataPoint.price ?? hoveredDataPoint.forecast}
              stroke="#94a3b8"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}

          {/* Render Price Data (Area or Candle) */}
          {type === 'area' && (
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="price"
              stroke={color}
              fillOpacity={1}
              fill={`url(#color${color})`}
              strokeWidth={2}
              activeDot={{ r: 6, strokeWidth: 0 }}
              isAnimationActive={true}
              animationDuration={300}
              animationEasing="ease-in-out"
            />
          )}

          {type === 'candle' && (
            <Bar
              yAxisId="right"
              dataKey="candleRange"
              shape={<Candle />}
              isAnimationActive={true}
              animationDuration={300}
              animationEasing="ease-in-out"
            />
          )}

          {/* Bollinger Bands */}
          {showBB && (
            <>
              <Area
                yAxisId="right"
                dataKey="bbUpper"
                stroke="#94a3b8"
                strokeWidth={1}
                strokeDasharray="3 3"
                fill="transparent"
                isAnimationActive={false}
                dot={false}
                activeDot={false}
              />
              <Area
                yAxisId="right"
                dataKey="bbLower"
                stroke="#94a3b8"
                strokeWidth={1}
                strokeDasharray="3 3"
                fill="#cbd5e1"
                fillOpacity={0.2}
                isAnimationActive={false}
                dot={false}
                activeDot={false}
              />
            </>
          )}

          {/* RSI Oscillator */}
          {showRSI && (
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="rsi"
              stroke="#7c3aed" // Violet 600
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          )}

          {/* MACD Oscillator */}
          {showMACD && (
            <>
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="macd"
                stroke="#2563eb" // Blue 600
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="macdSignal"
                stroke="#f97316" // Orange 500
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </>
          )}

          {/* Forecast Line */}
          {showForecast && (
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="forecast"
              stroke="#a855f7"
              strokeDasharray="5 5"
              strokeWidth={2}
              dot={false}
              activeDot={false} // Disable active dot to prevent interaction stealing
              connectNulls={true}
              name="Forecast"
              isAnimationActive={false}
            />
          )}

        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MarketChart;
