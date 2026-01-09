
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { CoinData, ForecastResult, HistoryPoint, NewsArticle, PriceAlert } from '../types';
import LightweightChart from '../components/LightweightChart';
import NewsCard from '../components/NewsCard';
import ArticleDetailModal from '../components/ArticleDetailModal';
import { Search, BellRing, Bell, Trash2, ArrowUp, ArrowDown, ChevronLeft, ChevronRight, GripHorizontal, Globe, BrainCircuit, Sparkles, RefreshCw, Zap, Target, ShieldAlert, Check, Calendar, Loader2, X, SlidersHorizontal, Activity, BarChart2, Star, ChevronDown, List, Hand, CheckCircle } from 'lucide-react';
import { getTopCoins, getHistoricalData, generateMarketForecast, getHistoricalNews, createSocketConnection } from '../services/apiService';

// Range Configuration for CryptoCompare
// Limit: Number of points. Aggregate: steps to combine. Type: minute/hour/day.
const getRangeParams = (range: string) => {
    switch (range) {
        // 1m: 60 x 1min candles = 1 hour
        case '1m': return { limit: 60, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 3m: 20 x 3min candles = 1 hour
        case '3m': return { limit: 20, aggregate: 3, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 5m: 72 x 5min candles = 6 hours
        case '5m': return { limit: 72, aggregate: 5, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 15m: 96 x 15min candles = 24 hours
        case '15m': return { limit: 96, aggregate: 15, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 30m: 48 x 30min candles = 24 hours
        case '30m': return { limit: 48, aggregate: 30, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 1h: 168 x 1hour candles = 7 days
        case '1H': return { limit: 168, aggregate: 1, type: 'hour' as const, format: { month: 'short', day: 'numeric', hour: '2-digit' } as const };
        // 4h: 180 x 4hour candles = 30 days
        case '4H': return { limit: 180, aggregate: 4, type: 'hour' as const, format: { month: 'short', day: 'numeric' } as const };
        // 1d: 90 x 1day candles = 3 months
        case '1D': return { limit: 90, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const };
        // 1w: 52 x 7day candles = 1 year
        case '1W': return { limit: 52, aggregate: 7, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // 1M: 12 x 30day candles = 1 year
        case '1M': return { limit: 12, aggregate: 30, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // 1Y: 365 x 1day candles = 1 year
        case '1Y': return { limit: 365, aggregate: 1, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // 5Y: 260 x 7day candles = 5 years (weekly candles)
        case '5Y': return { limit: 260, aggregate: 7, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // All: Max 2000 daily candles (~5.5 years)
        case 'All': return { limit: 2000, aggregate: 1, type: 'day' as const, format: { year: 'numeric' } as const };
        default: return { limit: 90, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const };
    }
};

type SortKey = 'price' | 'change' | 'percent';
type SortDirection = 'asc' | 'desc';


const Dashboard: React.FC = () => {
    const [coins, setCoins] = useState<CoinData[]>([]);
    const [selectedCoinSymbol, setSelectedCoinSymbol] = useState<string>('BTC');
    const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);
    const [loadingForecast, setLoadingForecast] = useState(false);
    const [combinedChartData, setCombinedChartData] = useState<any[]>([]);
    const [timeRange, setTimeRange] = useState('1D');
    const [loadingMarket, setLoadingMarket] = useState(true);

    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isCoinDropdownOpen, setIsCoinDropdownOpen] = useState(false);
    const sidebarRef = useRef<HTMLDivElement>(null);

    const [searchQuery, setSearchQuery] = useState('');
    const [favorites, setFavorites] = useState<Set<string>>(new Set(['BTC', 'ETH']));
    const [sortKey, setSortKey] = useState<SortKey>('price');
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

    // Custom Time Range & Hand Tool State
    const [isRangeSelecting, setIsRangeSelecting] = useState(false);
    const [rangeStart, setRangeStart] = useState<number | null>(null);
    const [chartVisibleRange, setChartVisibleRange] = useState<{ from: number; to: number } | null>(null);
    const [manualDateFrom, setManualDateFrom] = useState('');
    const [manualDateTo, setManualDateTo] = useState('');
    const [hasActiveRange, setHasActiveRange] = useState(false);

    const [chartType, setChartType] = useState<'area' | 'candle'>('candle');
    const [indicators, setIndicators] = useState({ rsi: false, macd: false, bollinger: false });
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const indicatorMenuRef = useRef<HTMLDivElement>(null);

    // Close indicator menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (indicatorMenuRef.current && !indicatorMenuRef.current.contains(event.target as Node)) {
                setShowIndicatorMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const [alertModalOpen, setAlertModalOpen] = useState(false);
    const [alertTargetPrice, setAlertTargetPrice] = useState<number>(0);
    const [alertSuccess, setAlertSuccess] = useState(false);
    const [priceAlerts, setPriceAlerts] = useState<PriceAlert[]>([
        { id: 'a1', symbol: 'BTC', targetPrice: 100000, condition: 'above', createdAt: Date.now() },
        { id: 'a2', symbol: 'ETH', targetPrice: 3200, condition: 'below', createdAt: Date.now() }
    ]);

    const [historyModalOpen, setHistoryModalOpen] = useState(false);
    const [historyDate, setHistoryDate] = useState<string>('');
    const [historicalNews, setHistoricalNews] = useState<NewsArticle[]>([]);
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [selectedHistoryArticle, setSelectedHistoryArticle] = useState<NewsArticle | null>(null);

    // WebSocket connection status and visual feedback
    const [wsStatus, setWsStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
    const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null);
    const [lastPrice, setLastPrice] = useState<number>(0);
    const lastPriceRef = useRef<number>(0);
    const throttleRef = useRef<{ timeout: NodeJS.Timeout | null; lastRun: number; pendingArg: any }>({
        timeout: null,
        lastRun: 0,
        pendingArg: null
    });

    const selectedCoin = coins.find(c => c.symbol === selectedCoinSymbol) || coins[0];

    useEffect(() => {
        const fetchMarket = async () => {
            setLoadingMarket(true);
            const data = await getTopCoins();
            setCoins(data);
            if (data.length > 0 && !data.find(c => c.symbol === selectedCoinSymbol)) {
                // If selected coin not in new list, default to first
                setSelectedCoinSymbol(data[0].symbol);
            }
            setLoadingMarket(false);
        };
        fetchMarket();
    }, []);



    // Update Chart Data (History + Forecast Merging)
    useEffect(() => {
        if (!selectedCoin) return;

        const fetchChart = async () => {
            const { limit, aggregate, type, format } = getRangeParams(timeRange);
            // CryptoCompare uses Symbol directly
            const history = await getHistoricalData(selectedCoin.symbol, limit, aggregate, type);

            let chartData = history.map(h => ({
                ...h,
                time: new Date(h.ts).toLocaleString('en-US', format)
            }));

            if (forecastResult && forecastResult.predictedPrices.length > 0) {
                const lastHistoryPoint = chartData[chartData.length - 1];
                const lastTime = lastHistoryPoint ? lastHistoryPoint.ts : new Date().getTime();
                const lastPrice = lastHistoryPoint ? lastHistoryPoint.price! : selectedCoin.price;

                // Simple Linear/Exponential Projection for visualized forecast
                // We use the time interval from the last two points to project future time
                const pointInterval = chartData.length > 1
                    ? chartData[chartData.length - 1].ts - chartData[chartData.length - 2].ts
                    : (type === 'minute' ? 600 * 1000 : 3600 * 1000); // Fallback

                const finalPredictedPrice = forecastResult.predictedPrices[forecastResult.predictedPrices.length - 1];
                const steps = forecastResult.predictedPrices.length;
                const ratio = finalPredictedPrice / lastPrice;
                const growthRate = Math.pow(ratio, 1 / steps) - 1;

                const futurePoints = Array.from({ length: steps }).map((_, index) => {
                    const futureTime = new Date(lastTime + (index + 1) * pointInterval);
                    const interpolatedPrice = lastPrice * Math.pow(1 + growthRate, index + 1);

                    return {
                        time: futureTime.toLocaleString('en-US', format),
                        ts: futureTime.getTime(),
                        forecast: parseFloat(interpolatedPrice.toFixed(2)),
                    };
                });

                const bridgePoint = {
                    ...chartData[chartData.length - 1],
                    forecast: lastPrice
                };
                chartData[chartData.length - 1] = bridgePoint;
                chartData = [...chartData, ...futurePoints];
            }
            setCombinedChartData(chartData);
        };

        fetchChart();
    }, [selectedCoin, forecastResult, timeRange]);

    // Real-time updates via WebSocket
    useEffect(() => {
        if (!selectedCoin) return;

        const socket = createSocketConnection('realtime');
        const symbol = selectedCoin.symbol;
        setWsStatus('connecting');

        socket.on('connect', () => {
            console.log('Connected to Realtime Socket');
            setWsStatus('connected');
            // Request to join room for kline and trade updates
            socket.emit('join-room', { symbol, type: 'kline' });
            socket.emit('join-room', { symbol, type: 'trade' });
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from Realtime Socket');
            setWsStatus('disconnected');
        });

        socket.on('reconnect', () => {
            console.log('Reconnected to Realtime Socket');
            setWsStatus('connected');
        });

        // Initialize price ref when coin changes
        if (selectedCoin) {
            lastPriceRef.current = selectedCoin.price;
        }

        // Robust Throttle Mechanism (Leading & Trailing)
        const processTradeUpdate = (message: any) => {
            const price = parseFloat(message.p);
            const currentLastPrice = lastPriceRef.current;

            // Trigger price flash animation
            if (currentLastPrice > 0) {
                if (price > currentLastPrice) {
                    setPriceFlash('up');
                } else if (price < currentLastPrice) {
                    setPriceFlash('down');
                }
                setTimeout(() => setPriceFlash(null), 300);
            }

            setLastPrice(price);
            lastPriceRef.current = price;

            throttleRef.current.lastRun = Date.now();
            throttleRef.current.pendingArg = null;
            throttleRef.current.timeout = null;


            setCombinedChartData(prevData => {
                if (prevData.length === 0) return prevData;

                const newData = [...prevData];

                // Find latest real candle
                let targetIndex = -1;
                for (let i = newData.length - 1; i >= 0; i--) {
                    if (newData[i].price !== undefined) {
                        targetIndex = i;
                        break;
                    }
                }
                if (targetIndex === -1) return prevData;

                const targetPoint = newData[targetIndex];

                // Update close price and high/low if needed
                newData[targetIndex] = {
                    ...targetPoint,
                    price: price,
                    high: Math.max(targetPoint.high || price, price),
                    low: Math.min(targetPoint.low || price, price),
                };

                return newData;
            });
        };

        const TRADE_THROTTLE_MS = 100;

        socket.on('trade', (message: any) => {
            const now = Date.now();
            const { lastRun, timeout } = throttleRef.current;

            if (now - lastRun >= TRADE_THROTTLE_MS) {
                // Leading edge: Execute immediately
                processTradeUpdate(message);
            } else {
                // Trailing edge: Schedule update
                throttleRef.current.pendingArg = message;

                if (!timeout) {
                    const delay = TRADE_THROTTLE_MS - (now - lastRun);
                    throttleRef.current.timeout = setTimeout(() => {
                        if (throttleRef.current.pendingArg) {
                            processTradeUpdate(throttleRef.current.pendingArg);
                        }
                    }, delay);
                }
            }
        });



        socket.on('kline', (payload: { symbol: string; data: any }) => {
            if (payload.symbol !== symbol) return;

            setCombinedChartData(prevData => {
                if (prevData.length === 0) return prevData;

                const newData = [...prevData];
                const lastPoint = newData[newData.length - 1];
                const kline = payload.data;
                const klineTime = new Date(kline.time);

                // Find index of the last real candle (has price)
                let targetIndex = -1;
                for (let i = newData.length - 1; i >= 0; i--) {
                    if (newData[i].price !== undefined) {
                        targetIndex = i;
                        break;
                    }
                }

                if (targetIndex === -1) targetIndex = newData.length - 1;

                const targetPoint = newData[targetIndex];

                // Debug log
                console.log('WS Update:', {
                    symbol,
                    klineTime: new Date(kline.time).toLocaleTimeString(),
                    targetTime: new Date(targetPoint.ts).toLocaleTimeString(),
                    klinePrice: kline.close,
                    currentPrice: targetPoint.price
                });

                // Update if within the same minute
                if (targetPoint && Math.abs(targetPoint.ts - kline.time) < 60000) {
                    newData[targetIndex] = {
                        ...targetPoint,
                        price: kline.close,
                        open: kline.open,
                        high: kline.high,
                        low: kline.low,
                        volume: kline.volume
                    };
                } else if (targetPoint && kline.time > targetPoint.ts) {
                    // Update the latest point to reflect live price even if it's a new candle
                    // Ideally we should push a new candle, but for simplicity in this view we ensure the "HEAD" moves.
                    newData[targetIndex] = {
                        ...targetPoint,
                        ts: kline.time,
                        time: new Date(kline.time).toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                        price: kline.close,
                        open: kline.open,
                        high: kline.high,
                        low: kline.low,
                        volume: kline.volume
                    };
                }

                return newData;
            });
        });

        return () => {
            socket.emit('leave-room', { symbol, type: 'kline' });
            socket.emit('leave-room', { symbol, type: 'trade' });
            socket.disconnect();
            setWsStatus('disconnected');
        };
    }, [selectedCoin?.symbol]);



    const handleForecast = async () => {
        setLoadingForecast(true);
        // Note: We do NOT clear the previous forecastResult here to ensure smooth transitions/polling
        if (!selectedCoin) {
            setLoadingForecast(false);
            return;
        }
        const trend = selectedCoin.change24h > 0 ? "Upward" : "Downward";
        try {
            const result = await generateMarketForecast(selectedCoin.name, trend, selectedCoin.price);
            setForecastResult(result);
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingForecast(false);
        }
    };

    // Clear forecast when coin changes (user must manually trigger new forecast)
    useEffect(() => {
        setForecastResult(null);
    }, [selectedCoinSymbol]);

    const processedCoins = useMemo(() => {
        let result = [...coins];

        if (searchQuery) {
            const q = searchQuery.toLowerCase();
            result = result.filter(c => c.name.toLowerCase().includes(q) || c.symbol.toLowerCase().includes(q));
        }
        result.sort((a, b) => {
            let valA = 0, valB = 0;
            switch (sortKey) {
                case 'price': valA = a.price; valB = b.price; break;
                case 'change': valA = a.change24h; valB = b.change24h; break;
                case 'percent': valA = (a.change24h / (a.price - a.change24h)); valB = (b.change24h / (b.price - b.change24h)); break;
            }
            return sortDirection === 'asc' ? valA - valB : valB - valA;
        });
        return result;
    }, [coins, searchQuery, favorites, sortKey, sortDirection]);

    const groupedAlerts = useMemo(() => {
        const groups: Record<string, PriceAlert[]> = {};
        priceAlerts.forEach(alert => {
            if (!groups[alert.symbol]) groups[alert.symbol] = [];
            groups[alert.symbol].push(alert);
        });
        return Object.entries(groups).map(([symbol, alerts]) => ({ symbol, alerts }));
    }, [priceAlerts]);

    const toggleFavorite = (e: React.MouseEvent, symbol: string) => {
        e.stopPropagation();
        const newFavs = new Set(favorites);
        if (newFavs.has(symbol)) newFavs.delete(symbol); else newFavs.add(symbol);
        setFavorites(newFavs);
    };

    const handleSort = (key: SortKey) => {
        if (sortKey === key) setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
        else { setSortKey(key); setSortDirection('desc'); }
    };

    const SortIcon = ({ colKey }: { colKey: SortKey }) => {
        if (sortKey !== colKey) return null;
        return sortDirection === 'asc' ? <ArrowUp size={10} className="inline ml-1" /> : <ArrowDown size={10} className="inline ml-1" />;
    };

    const headerStats = useMemo(() => {
        if (!selectedCoin) return { price: 0, change: 0, percent: 0 };

        // Use latest price from chart if available (real-time), otherwise fallback to initial API data
        let currentPrice = selectedCoin.price;
        if (combinedChartData.length > 0) {
            // Find the last "real" data point (not forecast only)
            const realPoints = combinedChartData.filter(p => p.price !== undefined);
            if (realPoints.length > 0) {
                currentPrice = realPoints[realPoints.length - 1].price!;
            }
        }

        if (combinedChartData.length === 0) {
            return {
                price: currentPrice,
                change: selectedCoin.change24h,
                percent: (selectedCoin.change24h / (selectedCoin.price - selectedCoin.change24h)) * 100
            };
        }

        // Calculate change: Current Price - Start Price of the view
        // Note: For "Change 24h", we ideally want the Close price from 24h ago. 
        // But for chart view context, we often show change relative to the chart period.
        // However, the UI label usually implies 24h change. 
        // If we want 24h change to update, we need to know the open/close of 24h ago even if chart is 1m.
        // Compromise: Update "current price" but keep "change" relative to the loaded chart range OR 
        // keep using selectedCoin.change24h but adjust it by the difference in current price.

        // Option A: Just update Price. Keep Change static (misleading).
        // Option B: Calculate dynamic change based on Chart Start (Good for 'Period Change').
        // The existing code calculated change from start of visible range. Let's stick to that but use dynamic currentPrice.

        const historicalPoints = combinedChartData.filter(p => p.forecast === undefined || (p.price !== undefined && p.forecast !== undefined));
        const startPrice = historicalPoints[0]?.price || currentPrice;
        const change = currentPrice - startPrice;
        const percent = startPrice !== 0 ? (change / startPrice) * 100 : 0;

        return { price: currentPrice, change, percent };
    }, [combinedChartData, selectedCoin]);

    const handleChartAlert = (price: number) => {
        setAlertTargetPrice(price);
        setAlertModalOpen(true);
        setAlertSuccess(false);
    };

    const handleChartHistory = async (date: string) => {
        setHistoryDate(date);
        setHistoryModalOpen(true);
        setLoadingHistory(true);
        setHistoricalNews([]);
        try {
            const news = await getHistoricalNews(selectedCoin.name, date);
            setHistoricalNews(news);
        } catch (e) { console.error(e); } finally { setLoadingHistory(false); }
    };

    const handleSaveAlert = () => {
        const condition = alertTargetPrice > selectedCoin.price ? 'above' : 'below';
        const newAlert: PriceAlert = { id: Date.now().toString(), symbol: selectedCoin.symbol, targetPrice: alertTargetPrice, condition: condition, createdAt: Date.now() };
        setPriceAlerts(prev => [newAlert, ...prev]);
        setAlertSuccess(true);
        setTimeout(() => { setAlertSuccess(false); setAlertModalOpen(false); }, 1500);
    };

    const deleteAlert = (id: string) => {
        setPriceAlerts(prev => prev.filter(a => a.id !== id));
    };

    const handleChartClick = (time: number) => {
        if (!isRangeSelecting) return;

        // Helper to format timestamp to datetime-local string (UTC Time to match chart)
        const formatDate = (ts: number) => {
            const d = new Date(ts * 1000);
            const pad = (n: number) => n.toString().padStart(2, '0');
            return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
        };

        if (rangeStart === null) {
            // First Click: Start Selection
            setRangeStart(time);
            setManualDateFrom(formatDate(time));
            setManualDateTo(''); // Clear End Date to indicate pending selection
        } else {
            // Second Click: End Selection
            const start = Math.min(rangeStart, time);
            const end = Math.max(rangeStart, time);

            // Update Visible Range
            setChartVisibleRange({ from: start, to: end });

            // Update Inputs with final sorted range
            setManualDateFrom(formatDate(start));
            setManualDateTo(formatDate(end));

            // Reset Mode
            setRangeStart(null);
            setIsRangeSelecting(false);
            setHasActiveRange(true);
            setTimeRange('Custom');
        }
    };

    const clearRangeSelection = () => {
        setChartVisibleRange(null);
        setManualDateFrom('');
        setManualDateTo('');
        setHasActiveRange(false);
        setIsRangeSelecting(false);
        setRangeStart(null);
    };

    const handleManualRangeApply = () => {
        if (manualDateFrom && manualDateTo) {
            // Parse as UTC (append 'Z' to treat as UTC time)
            const start = Math.floor(new Date(manualDateFrom + ':00Z').getTime() / 1000);
            const end = Math.floor(new Date(manualDateTo + ':00Z').getTime() / 1000);
            if (start < end) {
                setChartVisibleRange({ from: start, to: end });
                setHasActiveRange(true);
                setTimeRange('Custom');
            }
        }
    };

    if (loadingMarket || !selectedCoin) {
        return (
            <div className="h-[calc(100vh-3.5rem)] flex items-center justify-center bg-slate-50 dark:bg-slate-950">
                <Loader2 className="animate-spin text-indigo-600" size={32} />
            </div>
        );
    }

    return (
        <div className="h-[calc(100vh-3.5rem)] flex flex-col md:flex-row overflow-hidden bg-slate-50 dark:bg-slate-950 relative transition-colors duration-200">

            {/* LEFT: Main Chart Area */}
            <div className="flex-1 flex flex-col min-w-0 border-r border-slate-200 dark:border-slate-800">

                {/* Chart Header */}
                <div className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800">
                    <div className="p-5 space-y-4">
                        {/* Row 1: Title & Price */}
                        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
                            <div className="relative">
                                <button
                                    onClick={() => setIsCoinDropdownOpen(!isCoinDropdownOpen)}
                                    className="flex items-start gap-4 hover:bg-slate-50 dark:hover:bg-slate-800 p-2 -ml-2 rounded-xl transition-colors text-left group"
                                >
                                    <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-white shadow-sm ${selectedCoin.symbol === 'BTC' ? 'bg-orange-500' : selectedCoin.symbol === 'ETH' ? 'bg-blue-600' : selectedCoin.symbol === 'SOL' ? 'bg-purple-600' : 'bg-slate-700'}`}>
                                        {selectedCoin.symbol[0]}
                                    </div>

                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <h1 className="text-2xl font-bold text-slate-900 dark:text-white leading-none flex items-center gap-2">
                                                {selectedCoin.name} <span className="text-slate-400 font-normal text-lg">/ USD</span>
                                                <ChevronDown size={20} className={`text-slate-400 transition-transform ${isCoinDropdownOpen ? 'rotate-180' : ''}`} />
                                            </h1>
                                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 border transition-all ${wsStatus === 'connected'
                                                ? 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
                                                : 'bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800'
                                                }`}>
                                                <span className={`w-1.5 h-1.5 rounded-full ${wsStatus === 'connected' ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
                                                {wsStatus === 'connected' ? 'LIVE' : 'CONNECTING'}
                                            </span>
                                        </div>
                                        <div className="flex items-baseline gap-3">
                                            <span className={`text-4xl font-bold text-slate-900 dark:text-white tracking-tight transition-all duration-300 ${priceFlash}`}>
                                                ${headerStats.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                            </span>
                                            <div className={`flex items-center gap-1 font-bold text-lg ${headerStats.change >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                {headerStats.change > 0 ? '+' : ''}{headerStats.change.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({Math.abs(headerStats.percent).toFixed(2)}%)
                                                <span className="text-sm text-slate-400 dark:text-slate-500 ml-2 font-normal">{timeRange === 'All' ? 'Max' : timeRange}</span>
                                            </div>
                                        </div>
                                    </div>
                                </button>

                                {/* COIN DROPDOWN MENU */}
                                {isCoinDropdownOpen && (
                                    <div className="absolute top-full left-0 mt-2 w-96 bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-800 z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[600px]">
                                        {/* Dropdown Header & Search */}
                                        <div className="p-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50">
                                            <div className="relative">
                                                <Search size={16} className="absolute left-3 top-2.5 text-slate-400" />
                                                <input
                                                    type="text"
                                                    placeholder="Search coin..."
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                    className="w-full pl-9 pr-4 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20"
                                                    autoFocus
                                                />
                                            </div>
                                            <div className="flex justify-between items-center px-2 pt-3 text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                                                <div className="w-[30%]">Asset</div>
                                                <div className="w-[25%] text-right cursor-pointer hover:text-indigo-600" onClick={() => handleSort('price')}>Price <SortIcon colKey="price" /></div>
                                                <div className="w-[20%] text-right cursor-pointer hover:text-indigo-600" onClick={() => handleSort('change')}>Chg <SortIcon colKey="change" /></div>
                                                <div className="w-[20%] text-right cursor-pointer hover:text-indigo-600" onClick={() => handleSort('percent')}>% <SortIcon colKey="percent" /></div>
                                            </div>
                                        </div>

                                        {/* Dropdown List */}
                                        <div className="overflow-y-auto custom-scrollbar p-2">
                                            {processedCoins.length === 0 ? (
                                                <div className="text-center py-8 text-slate-400 text-sm">No coins found</div>
                                            ) : (
                                                processedCoins.map(coin => (
                                                    <button
                                                        key={coin.symbol}
                                                        onClick={() => {
                                                            setSelectedCoinSymbol(coin.symbol);
                                                            setForecastResult(null);
                                                            setIsCoinDropdownOpen(false);
                                                        }}
                                                        className={`w-full flex items-center justify-between p-2.5 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors group ${selectedCoinSymbol === coin.symbol ? 'bg-indigo-50 dark:bg-indigo-900/20' : ''}`}
                                                    >
                                                        <div className="flex items-center gap-3 w-[30%]">
                                                            <div onClick={(e) => { e.stopPropagation(); toggleFavorite(e, coin.symbol); }} className="text-slate-300 hover:text-amber-400">
                                                                <Star size={14} fill={favorites.has(coin.symbol) ? "#fbbf24" : "none"} className={favorites.has(coin.symbol) ? "text-amber-400" : ""} />
                                                            </div>
                                                            <div className="text-left">
                                                                <div className="font-bold text-sm text-slate-900 dark:text-white leading-none">{coin.symbol}</div>
                                                                <div className="text-[10px] text-slate-400">{coin.name}</div>
                                                            </div>
                                                        </div>
                                                        <div className="w-[25%] text-right text-sm font-medium text-slate-700 dark:text-slate-300">
                                                            ${coin.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                                        </div>
                                                        <div className={`w-[20%] text-right text-xs ${coin.change24h >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                                                            {coin.change24h > 0 ? '+' : ''}{Math.abs(coin.change24h).toFixed(2)}
                                                        </div>
                                                        <div className={`w-[20%] text-right`}>
                                                            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${coin.change24h >= 0 ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-500/10 dark:text-rose-400'}`}>
                                                                {((coin.change24h / (coin.price - coin.change24h)) * 100).toFixed(2)}%
                                                            </span>
                                                        </div>
                                                    </button>
                                                ))
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Right Side: Sidebar Toggle (Polished) */}
                            <div className="flex gap-2 self-start">
                                <button
                                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                                    className="flex items-center justify-center w-8 h-8 rounded-full bg-white dark:bg-slate-800 shadow-md border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-50 dark:hover:bg-slate-700 transition-all transform hover:scale-105 active:scale-95"
                                    title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
                                >
                                    {isSidebarOpen ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
                                </button>
                            </div>
                        </div>

                        {/* Row 2: Controls (Time Range & Indicators) */}
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2">
                            {/* Time Range Buttons - 2 Rows */}
                            <div className="flex flex-col gap-1">
                                {/* Row 1: Short Intervals (Minutes & Hours) */}
                                <div className="flex gap-1">
                                    {['1m', '3m', '5m', '15m', '30m', '1H', '4H'].map((range) => (
                                        <button
                                            key={range}
                                            onClick={() => setTimeRange(range)}
                                            className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${timeRange === range
                                                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20'
                                                : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 dark:text-slate-400'
                                                }`}
                                        >    {range}
                                        </button>
                                    ))}
                                </div>
                                {/* Row 2: Long Intervals (Days, Weeks, Months, Years) */}
                                <div className="flex gap-1">
                                    {['1D', '1W', '1M', '1Y', '5Y', 'All'].map((range) => (
                                        <button
                                            key={range}
                                            onClick={() => setTimeRange(range)}
                                            className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${timeRange === range
                                                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20'
                                                : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 dark:text-slate-400'
                                                }`}
                                        >    {range}
                                        </button>

                                    ))}
                                </div>
                            </div>

                            {/* Custom Range Inputs & Hand Tool */}
                            <div className="flex items-center gap-2">
                                <div className="hidden lg:flex items-center gap-2 bg-slate-100 dark:bg-slate-800/50 p-1 rounded-lg border border-slate-200 dark:border-slate-800/50">
                                    <input
                                        type="datetime-local"
                                        value={manualDateFrom}
                                        onChange={(e) => setManualDateFrom(e.target.value)}
                                        className="bg-transparent text-[10px] font-medium text-slate-600 dark:text-slate-300 focus:outline-none w-36 px-1"
                                    />
                                    <span className="text-slate-400 text-xs">-</span>
                                    <input
                                        type="datetime-local"
                                        value={manualDateTo}
                                        onChange={(e) => setManualDateTo(e.target.value)}
                                        className="bg-transparent text-[10px] font-medium text-slate-600 dark:text-slate-300 focus:outline-none w-36 px-1"
                                    />
                                    <button onClick={handleManualRangeApply}
                                        className="p-1 hover:bg-white dark:hover:bg-slate-700 rounded transition-colors text-indigo-600 dark:text-indigo-400"
                                    >
                                        <Check size={12} />
                                    </button>
                                    <span className="text-[9px] text-slate-400 dark:text-slate-500 font-medium px-1">UTC</span>
                                </div>

                                <button
                                    onClick={() => {
                                        if (hasActiveRange) {
                                            clearRangeSelection();
                                        } else {
                                            setIsRangeSelecting(!isRangeSelecting);
                                            setRangeStart(null);
                                        }
                                    }}
                                    className={`p-1.5 rounded-lg border transition-all ${hasActiveRange
                                        ? 'bg-rose-50 dark:bg-rose-900/20 text-rose-600 dark:text-rose-400 border-rose-200 dark:border-rose-800 hover:bg-rose-100 dark:hover:bg-rose-900/30'
                                        : isRangeSelecting
                                            ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800 ring-2 ring-indigo-500/20'
                                            : 'bg-white dark:bg-slate-900 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
                                        }`}
                                    title={hasActiveRange ? "Clear range selection" : "Hand Tool: Click two points on chart to select range"}
                                >
                                    {hasActiveRange ? <X size={16} /> : <Hand size={16} className={isRangeSelecting ? "animate-pulse" : ""} />}
                                </button>
                            </div>

                            <div className="flex items-center gap-2">
                                {/* Chart Type Toggles */}
                                <div className="hidden md:flex bg-slate-100 dark:bg-slate-800/50 p-1 rounded-lg border border-slate-200 dark:border-slate-800/50">
                                    <button
                                        onClick={() => setChartType('area')}
                                        className={`p-1.5 rounded-md transition-all ${chartType === 'area' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                        title="Area Chart"
                                    >
                                        <Activity size={16} />
                                    </button>
                                    <button
                                        onClick={() => setChartType('candle')}
                                        className={`p-1.5 rounded-md transition-all ${chartType === 'candle' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                        title="Candlestick Chart"
                                    >
                                        <BarChart2 size={16} />
                                    </button>
                                </div>

                                <div className="h-4 w-px bg-slate-200 dark:bg-slate-700 mx-1"></div>

                                {/* Indicators Menu */}
                                <div className="relative" ref={indicatorMenuRef}>
                                    <button
                                        onClick={() => setShowIndicatorMenu(!showIndicatorMenu)}
                                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg border transition-all ${showIndicatorMenu
                                            ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800'
                                            : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800'
                                            }`}
                                    >
                                        <SlidersHorizontal size={14} />
                                        Indicators
                                        <ChevronDown size={12} className={`transition-transform ${showIndicatorMenu ? 'rotate-180' : ''}`} />
                                    </button>

                                    {/* Dropdown Menu */}
                                    {showIndicatorMenu && (
                                        <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-slate-900 rounded-xl shadow-xl border border-slate-200 dark:border-slate-800 z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                                            <div className="p-2 space-y-1">
                                                <button
                                                    onClick={() => setIndicators(prev => ({ ...prev, rsi: !prev.rsi }))}
                                                    className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 rounded-lg group"
                                                >
                                                    <div className="flex items-center gap-2">
                                                        {indicators.rsi && <Check size={14} className="text-indigo-600" />}
                                                        <span>RSI</span>
                                                    </div>
                                                </button>
                                                <button
                                                    onClick={() => setIndicators(prev => ({ ...prev, macd: !prev.macd }))}
                                                    className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 rounded-lg group"
                                                >
                                                    <div className="flex items-center gap-2">
                                                        {indicators.macd && <Check size={14} className="text-indigo-600" />}
                                                        <span>MACD</span>
                                                    </div>
                                                </button>
                                                <button
                                                    onClick={() => setIndicators(prev => ({ ...prev, bollinger: !prev.bollinger }))}
                                                    className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 rounded-lg group"
                                                >
                                                    <div className="flex items-center gap-2">
                                                        {indicators.bollinger && <Check size={14} className="text-indigo-600" />}
                                                        <span>Bollinger Bands</span>
                                                    </div>
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chart Content */}
                <div className="flex-1 relative bg-white dark:bg-slate-900">
                    <LightweightChart
                        data={combinedChartData}
                        color={headerStats.change >= 0 ? '#10b981' : '#f43f5e'}
                        type={chartType}
                        indicators={indicators}
                        onChartClick={handleChartClick}
                        visibleRange={chartVisibleRange}
                    />
                    {isRangeSelecting && (
                        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-indigo-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg z-10 pointer-events-none animate-in fade-in slide-in-from-top-2">
                            {rangeStart ? "Click end point..." : "Click start point..."}
                        </div>
                    )}
                </div>
            </div>



            {/* RIGHT: Sidebar Panel (Market Intelligence) */}
            {
                isSidebarOpen && (
                    <div
                        ref={sidebarRef}
                        className="w-full md:w-80 lg:w-96 flex flex-col bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800"
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
                            <h3 className="text-sm font-bold text-slate-800 dark:text-white uppercase flex items-center gap-2">
                                <Zap size={16} className="text-indigo-500" />
                                Market Intelligence
                            </h3>
                            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                AI-powered analysis & news for {selectedCoin.name}
                            </p>
                        </div>

                        {/* Content Area */}
                        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-slate-50 dark:bg-slate-950">
                            {forecastResult ? (
                                <div className="space-y-4 animate-in fade-in">

                                    {/* Action Recommendation Card */}
                                    {forecastResult.recommendation && (
                                        <div className="bg-gradient-to-br from-indigo-50 to-white dark:from-slate-800 dark:to-slate-900 p-4 rounded-xl border border-indigo-100 dark:border-slate-700 shadow-sm">
                                            <div className="flex justify-between items-center mb-3">
                                                <h4 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wide flex items-center gap-1">
                                                    <Target size={14} /> Trade Signal
                                                </h4>
                                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${forecastResult.recommendation.action === 'Buy' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                                                    forecastResult.recommendation.action === 'Sell' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                                                        'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                                                    }`}>
                                                    {forecastResult.recommendation.action}
                                                </span>
                                            </div>
                                            <div className="grid grid-cols-2 gap-2 text-xs">
                                                <div className="bg-white dark:bg-slate-800 p-2 rounded border border-slate-100 dark:border-slate-700">
                                                    <span className="text-[10px] text-slate-500 block mb-0.5">Entry Zone</span>
                                                    <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">{forecastResult.recommendation.entryZone}</span>
                                                </div>
                                                <div className="bg-white dark:bg-slate-800 p-2 rounded border border-slate-100 dark:border-slate-700">
                                                    <span className="text-[10px] text-slate-500 block mb-0.5">Target</span>
                                                    <span className="font-mono font-semibold text-green-600 dark:text-green-400">{forecastResult.recommendation.targetPrice}</span>
                                                </div>
                                            </div>
                                            <div className="mt-2 flex items-center gap-2 text-[10px] text-red-500 bg-red-50 dark:bg-red-900/20 px-2 py-1.5 rounded border border-red-100 dark:border-red-900/30">
                                                <ShieldAlert size={10} />
                                                <span>Stop Loss: {forecastResult.recommendation.stopLoss}</span>
                                            </div>
                                        </div>
                                    )}

                                    {/* Confidence */}
                                    <div className="bg-white dark:bg-slate-900 p-3 rounded-lg border border-slate-200 dark:border-slate-800 shadow-sm">
                                        <span className="text-xs text-slate-500 dark:text-slate-400">Confidence</span>
                                        <div className="flex items-center gap-3 mb-2">
                                            <div className="flex-1 h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                                                <div className={`h-full rounded-full ${forecastResult.confidenceScore > 70 ? 'bg-green-500' : 'bg-yellow-500'}`} style={{ width: `${forecastResult.confidenceScore}%` }}></div>
                                            </div>
                                            <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{forecastResult.confidenceScore}%</span>
                                        </div>
                                        {/* Reasoning */}
                                        <div>
                                            <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 mb-2 uppercase">Analysis</h4>
                                            <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                                                {forecastResult.reasoning}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Sources */}
                                    {forecastResult.sources && forecastResult.sources.length > 0 && (
                                        <div>
                                            <h4 className="text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 uppercase flex items-center gap-1">
                                                <Globe size={10} /> Grounding
                                            </h4>
                                            <div className="space-y-2">
                                                {forecastResult.sources.map((s, i) => (
                                                    <a key={i} href={s.url} target="_blank" className="block p-2 rounded bg-white dark:bg-slate-900 border border-slate-100 dark:border-slate-800 hover:border-indigo-200 dark:hover:border-indigo-900 transition-colors group">
                                                        <div className="text-xs font-medium text-slate-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 mb-1 line-clamp-2">
                                                            {s.title}
                                                        </div>
                                                        <div className="text-[10px] text-slate-400 flex items-center gap-1">
                                                            <Globe size={8} />
                                                            {new URL(s.url).hostname.replace('www.', '')}
                                                        </div>
                                                    </a>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    <button
                                        onClick={handleForecast}
                                        className="w-full py-2 mt-4 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-indigo-600 dark:text-indigo-400 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 shadow-sm"
                                    >
                                        <RefreshCw size={14} /> Refresh Analysis
                                    </button>
                                </div>
                            ) : loadingForecast ? (
                                // Skeleton Loading State
                                <div className="space-y-4 animate-pulse">
                                    <div className="h-40 bg-slate-200 dark:bg-slate-800 rounded-xl"></div>
                                    <div className="h-24 bg-slate-200 dark:bg-slate-800 rounded-xl"></div>
                                    <div className="space-y-2">
                                        <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-3/4"></div>
                                        <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/2"></div>
                                    </div>
                                </div>
                            ) : (
                                // No forecast - show generate button
                                <div className="flex flex-col items-center justify-center py-20 px-4 text-center">
                                    <div className="w-16 h-16 bg-indigo-50 dark:bg-indigo-900/20 rounded-full flex items-center justify-center mb-4">
                                        <BrainCircuit size={32} className="text-indigo-500" />
                                    </div>
                                    <h3 className="font-bold text-slate-900 dark:text-white mb-2">Market Intelligence</h3>
                                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                                        Generate real-time AI technical analysis and news insights for {selectedCoin.name}.
                                    </p>
                                    <button
                                        onClick={handleForecast}
                                        className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-bold transition-all shadow-lg shadow-indigo-500/30 flex items-center gap-2 transform hover:scale-105"
                                    >
                                        <Sparkles size={18} />
                                        Generate AI Forecast
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )
            }

            {/* -- MODALS -- */}

            {/* Alert Modal */}
            {
                alertModalOpen && (
                    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-in fade-in">
                        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 rounded-2xl w-full max-w-sm shadow-2xl relative animate-in zoom-in-95 duration-200">
                            <button onClick={() => setAlertModalOpen(false)} className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"><X size={20} /></button>

                            {alertSuccess ? (
                                <div className="flex flex-col items-center justify-center py-8 text-green-600">
                                    <CheckCircle size={48} className="mb-4" />
                                    <h3 className="text-lg font-bold text-slate-900 dark:text-white">Alert Confirmed!</h3>
                                    <p className="text-slate-500 dark:text-slate-400 text-sm">We'll notify you when price hits ${alertTargetPrice.toLocaleString()}</p>
                                </div>
                            ) : (
                                <>
                                    <div className="flex items-center gap-2 mb-4 text-slate-900 dark:text-white">
                                        <Bell className="text-indigo-600 dark:text-indigo-400" />
                                        <h3 className="font-bold text-lg">Set Chart Alert</h3>
                                    </div>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">Asset</label>
                                            <div className="w-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-2 text-slate-900 dark:text-white text-sm font-medium">
                                                {selectedCoin.name} ({selectedCoin.symbol})
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">Target Price ($)</label>
                                            <input type="number" value={alertTargetPrice} onChange={(e) => setAlertTargetPrice(parseFloat(e.target.value))} className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-2 text-slate-900 dark:text-white text-sm outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500" />
                                        </div>
                                        <button onClick={handleSaveAlert} className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-colors">Save Alert</button>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                )
            }

            {/* Historical News Modal */}
            {
                historyModalOpen && (
                    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-in fade-in">
                        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl relative animate-in zoom-in-95 duration-200">
                            <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between bg-white dark:bg-slate-900 rounded-t-2xl z-10">
                                <div className="flex items-center gap-2">
                                    <Calendar className="text-indigo-600 dark:text-indigo-400" size={20} />
                                    <div>
                                        <h3 className="font-bold text-slate-900 dark:text-white">Historical Context</h3>
                                        <p className="text-xs text-slate-500">News around {historyDate.split(',')[0]}</p>
                                    </div>
                                </div>
                                <button onClick={() => setHistoryModalOpen(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"><X size={20} /></button>
                            </div>

                            <div className="flex-1 overflow-y-auto p-6 bg-slate-50 dark:bg-slate-950 rounded-b-2xl custom-scrollbar">
                                {loadingHistory ? (
                                    <div className="flex flex-col items-center justify-center h-64 space-y-3">
                                        <Loader2 className="animate-spin text-indigo-600 dark:text-indigo-400" size={32} />
                                        <p className="text-sm text-slate-500">Searching archives...</p>
                                    </div>
                                ) : historicalNews.length === 0 ? (
                                    <div className="text-center py-20 text-slate-500 text-sm">No specific news found for this exact timeframe.</div>
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {historicalNews.map((news) => (
                                            <NewsCard key={news.id} article={news} onClick={(article) => setSelectedHistoryArticle(article)} />
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )
            }


            {/* Detail Article Modal for History */}
            {
                selectedHistoryArticle && (
                    <ArticleDetailModal article={selectedHistoryArticle} onClose={() => setSelectedHistoryArticle(null)} />
                )
            }
        </div >
    );
};

export default Dashboard;
