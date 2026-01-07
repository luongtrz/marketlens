import React, { useState, useEffect, useRef, useMemo } from 'react';
import { CoinData, ForecastResult, HistoryPoint, NewsArticle, PriceAlert } from '../types';
import MarketChart from '../components/MarketChart';
import NewsCard from '../components/NewsCard';
import ArticleDetailModal from '../components/ArticleDetailModal';
import { RefreshCw, Zap, Search, BarChart2, TrendingUp, Globe, List, Loader2, Layers, Check, GripHorizontal, PanelRightClose, PanelRightOpen, Star, ArrowUp, ArrowDown, X, Bell, CheckCircle, Calendar, Trash2, BellRing, ChevronLeft, ChevronRight, Target, ShieldAlert, ArrowRight, BrainCircuit, Sparkles } from 'lucide-react';
import { getTopCoins, getHistoricalData, generateMarketForecast, getHistoricalNews, createSocketConnection } from '../services/apiService';

// Range Configuration for CryptoCompare
// Limit: Number of points. Aggregate: steps to combine. Type: minute/hour/day.
const getRangeParams = (range: string) => {
    switch (range) {
        // 30 Minutes: 30 minutes, 1-min candles
        case '30m': return { limit: 30, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 1 Hour: 60 minutes, 1-min candles
        case '1H': return { limit: 60, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 12 Hours: 720 minutes. 5-min candles => 144 points
        case '12H': return { limit: 144, aggregate: 5, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 1 Minute view: Last 60 candles (1 hour of 1-minute candles)
        case '1m': return { limit: 60, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 24 Hours. Limit 1440 if we want full minute data or 144 points with aggregate 10
        case '1D': return { limit: 144, aggregate: 10, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
        // 1 Month = 30 days = 720 hours. Limit 720, type hour
        case '1M': return { limit: 720, aggregate: 1, type: 'hour' as const, format: { month: 'short', day: 'numeric' } as const };
        // 3 Months = 90 days. Limit 90, type day
        case '3M': return { limit: 90, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const };
        // 1 Year = 365 days. Limit 365, type day
        case '1Y': return { limit: 365, aggregate: 1, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // 5 Years = 1825 days. Limit 1825, type day. limit max is 2000.
        case '5Y': return { limit: 1825, aggregate: 1, type: 'day' as const, format: { month: 'short', year: '2-digit' } as const };
        // All Time: Just get max daily data (2000 days is ~5.5 years, usually enough for most recent crypto context)
        case 'All': return { limit: 2000, aggregate: 1, type: 'day' as const, format: { month: 'short', year: 'numeric' } as const };
        default: return { limit: 144, aggregate: 10, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const };
    }
};

type SortKey = 'price' | 'change' | 'percent';
type SortDirection = 'asc' | 'desc';
type WatchlistTab = 'all' | 'favs' | 'alerts';

const Dashboard: React.FC = () => {
    const [coins, setCoins] = useState<CoinData[]>([]);
    const [selectedCoinSymbol, setSelectedCoinSymbol] = useState<string>('BTC');
    const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);
    const [loadingForecast, setLoadingForecast] = useState(false);
    const [combinedChartData, setCombinedChartData] = useState<any[]>([]);
    const [chartType, setChartType] = useState<'area' | 'candle'>('candle');
    const [timeRange, setTimeRange] = useState('1D');
    const [loadingMarket, setLoadingMarket] = useState(true);

    const [showRSI, setShowRSI] = useState(false);
    const [showMACD, setShowMACD] = useState(false);
    const [showBB, setShowBB] = useState(false);
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const indicatorMenuRef = useRef<HTMLDivElement>(null);

    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [watchlistHeight, setWatchlistHeight] = useState(50);
    const [isResizing, setIsResizing] = useState(false);
    const sidebarRef = useRef<HTMLDivElement>(null);

    const [searchQuery, setSearchQuery] = useState('');
    const [favorites, setFavorites] = useState<Set<string>>(new Set(['BTC', 'ETH']));
    const [activeTab, setActiveTab] = useState<WatchlistTab>('all');
    const [sortKey, setSortKey] = useState<SortKey>('price');
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

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

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (indicatorMenuRef.current && !indicatorMenuRef.current.contains(event.target as Node)) {
                setShowIndicatorMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
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

        const socket = createSocketConnection();
        const symbol = selectedCoin.symbol;

        socket.on('connect', () => {
            console.log('Connected to Realtime Socket');
            // Request to join room for kline and trade updates
            socket.emit('join-room', { symbol, type: 'kline' });
            socket.emit('join-room', { symbol, type: 'trade' });
        });


        // Throttle trade updates to avoid excessive re-renders
        let lastTradeUpdate = 0;
        const TRADE_THROTTLE_MS = 200; // Update max 5 times/sec

        socket.on('trade', (message: any) => {
            // message format: { e: 'trade', s: 'BTCUSDT', p: '91234.56', ... }
            const now = Date.now();
            if (now - lastTradeUpdate < TRADE_THROTTLE_MS) return;
            lastTradeUpdate = now;

            const price = parseFloat(message.p);
            console.log('Trade Update:', { price, symbol: message.s });

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
        };
    }, [selectedCoin?.symbol]);

    // Sidebar Resizing Logic
    const startResizing = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    };
    const stopResizing = () => setIsResizing(false);
    const resize = (e: MouseEvent) => {
        if (isResizing && sidebarRef.current) {
            const sidebarRect = sidebarRef.current.getBoundingClientRect();
            const newHeight = e.clientY - sidebarRect.top;
            const percentage = (newHeight / sidebarRect.height) * 100;
            const clamped = Math.min(Math.max(percentage, 20), 80);
            setWatchlistHeight(clamped);
        }
    };

    useEffect(() => {
        if (isResizing) {
            window.addEventListener('mousemove', resize);
            window.addEventListener('mouseup', stopResizing);
        } else {
            window.removeEventListener('mousemove', resize);
            window.removeEventListener('mouseup', stopResizing);
        }
        return () => {
            window.removeEventListener('mousemove', resize);
            window.removeEventListener('mouseup', stopResizing);
        };
    }, [isResizing]);

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
        if (activeTab === 'favs') result = result.filter(c => favorites.has(c.symbol));
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
    }, [coins, searchQuery, favorites, activeTab, sortKey, sortDirection]);

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
        setTimeout(() => { setAlertSuccess(false); setAlertModalOpen(false); setActiveTab('alerts'); }, 1500);
    };

    const deleteAlert = (id: string) => {
        setPriceAlerts(prev => prev.filter(a => a.id !== id));
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
                            <div className="flex items-start gap-4">
                                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-white shadow-sm ${selectedCoin.symbol === 'BTC' ? 'bg-orange-500' : selectedCoin.symbol === 'ETH' ? 'bg-blue-600' : selectedCoin.symbol === 'SOL' ? 'bg-purple-600' : 'bg-slate-700'}`}>
                                    {selectedCoin.symbol[0]}
                                </div>

                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <h1 className="text-2xl font-bold text-slate-900 dark:text-white leading-none">{selectedCoin.name} / U.S. Dollar</h1>
                                        <span className="bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1 border border-emerald-200 dark:border-emerald-800">
                                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse"></span>
                                            Live
                                        </span>
                                    </div>
                                    <div className="flex items-baseline gap-3">
                                        <span className="text-4xl font-bold text-slate-900 dark:text-white tracking-tight">
                                            ${headerStats.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                        </span>
                                        <span className={`text-lg font-bold flex items-center ${headerStats.change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                                            {headerStats.change > 0 ? '+' : ''}{headerStats.change.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({Math.abs(headerStats.percent).toFixed(2)}%)
                                            <span className="text-sm text-slate-400 dark:text-slate-500 ml-2 font-normal">{timeRange === 'All' ? 'Max' : timeRange}</span>
                                        </span>
                                    </div>
                                </div>
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
                            <div className="flex gap-1">
                                {['1m', '30m', '1H', '12H', '1D', '1M', '3M', '1Y', '5Y', 'All'].map((range) => (
                                    <button
                                        key={range}
                                        onClick={() => setTimeRange(range)}
                                        className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${timeRange === range ? 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm border border-slate-200 dark:border-slate-700' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                                    >
                                        {range}
                                    </button>
                                ))}
                            </div>

                            <div className="flex items-center gap-2">
                                <div className="relative" ref={indicatorMenuRef}>
                                    <button
                                        onClick={() => setShowIndicatorMenu(!showIndicatorMenu)}
                                        className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${showIndicatorMenu || showRSI || showMACD || showBB ? 'bg-indigo-50 dark:bg-slate-800 text-indigo-600 dark:text-indigo-400' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'}`}
                                    >
                                        <Layers size={14} />
                                        <span className="hidden sm:inline">Indicators</span>
                                    </button>

                                    {showIndicatorMenu && (
                                        <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl py-1 z-50 animate-in fade-in zoom-in-95 duration-200">
                                            <div className="px-3 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider">Overlays</div>
                                            <button
                                                onClick={() => setShowBB(!showBB)}
                                                className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                                            >
                                                <span>Bollinger Bands</span>
                                                {showBB && <Check size={14} className="text-indigo-600 dark:text-indigo-400" />}
                                            </button>

                                            <div className="px-3 py-2 mt-1 text-xs font-bold text-slate-400 uppercase tracking-wider border-t border-slate-100 dark:border-slate-800">Oscillators</div>
                                            <button
                                                onClick={() => setShowRSI(!showRSI)}
                                                className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                                            >
                                                <span>RSI (14)</span>
                                                {showRSI && <Check size={14} className="text-indigo-600 dark:text-indigo-400" />}
                                            </button>
                                            <button
                                                onClick={() => setShowMACD(!showMACD)}
                                                className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                                            >
                                                <span>MACD</span>
                                                {showMACD && <Check size={14} className="text-indigo-600 dark:text-indigo-400" />}
                                            </button>
                                        </div>
                                    )}
                                </div>

                                <div className="flex items-center gap-1 border-l border-slate-200 dark:border-slate-800 pl-4 ml-2">
                                    <button onClick={() => setChartType('area')} className={`p-1.5 rounded transition-all ${chartType === 'area' ? 'bg-slate-100 dark:bg-slate-800 text-indigo-600 dark:text-indigo-400' : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'}`} title="Line">
                                        <TrendingUp size={16} />
                                    </button>
                                    <button onClick={() => setChartType('candle')} className={`p-1.5 rounded transition-all ${chartType === 'candle' ? 'bg-slate-100 dark:bg-slate-800 text-indigo-600 dark:text-indigo-400' : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'}`} title="Candles">
                                        <BarChart2 size={16} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chart Content */}
                <div className="flex-1 relative bg-white dark:bg-slate-900">
                    <MarketChart
                        data={combinedChartData}
                        color={headerStats.change >= 0 ? '#10b981' : '#f43f5e'}
                        type={chartType}
                        showRSI={showRSI}
                        showMACD={showMACD}
                        showBB={showBB}
                        onTimeRangeChange={setTimeRange}
                        onAddAlert={handleChartAlert}
                        onAnalyzeHistory={handleChartHistory}
                    />
                </div>
            </div>

            {/* RIGHT: Sidebar Panel (Collapsible & Resizable) */}
            {isSidebarOpen && (
                <div
                    ref={sidebarRef}
                    className="w-full md:w-80 lg:w-96 flex flex-col bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800"
                >

                    {/* SECTION 1: Watchlist */}
                    <div style={{ height: `${watchlistHeight}%` }} className="flex flex-col border-b border-slate-200 dark:border-slate-800 min-h-[180px]">
                        {/* Watchlist Header */}
                        <div className="p-3 border-b border-slate-200 dark:border-slate-800 space-y-3 bg-white dark:bg-slate-900">
                            <div className="flex items-center justify-between">
                                <h3 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase flex items-center gap-2">
                                    {activeTab === 'alerts' ? <BellRing size={14} /> : <List size={14} />}
                                    {activeTab === 'alerts' ? 'Active Alerts' : 'Watchlist'}
                                </h3>
                                {/* Tab Switcher */}
                                <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-0.5">
                                    <button
                                        onClick={() => setActiveTab('all')}
                                        className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${activeTab === 'all' ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-white' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                    >
                                        All
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('favs')}
                                        className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${activeTab === 'favs' ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-white' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                    >
                                        Favs
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('alerts')}
                                        className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${activeTab === 'alerts' ? 'bg-white dark:bg-slate-700 shadow-sm text-slate-900 dark:text-white' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                    >
                                        Alerts
                                    </button>
                                </div>
                            </div>

                            {/* Search Bar */}
                            <div className="relative">
                                <Search size={14} className="absolute left-2.5 top-2 text-slate-400" />
                                <input
                                    type="text"
                                    placeholder={activeTab === 'alerts' ? "Search alerts..." : "Search coin..."}
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:border-indigo-500 text-slate-900 dark:text-slate-100 transition-colors placeholder-slate-400 dark:placeholder-slate-500"
                                />
                            </div>

                            {activeTab !== 'alerts' && (
                                <div className="flex justify-between items-center px-3 pt-1 text-[10px] text-slate-400 dark:text-slate-500 font-semibold uppercase tracking-wider">
                                    <div className="w-[30%] text-left">Asset</div>
                                    <div className="w-[25%] text-right cursor-pointer hover:text-slate-600 dark:hover:text-slate-300" onClick={() => handleSort('price')}>Price <SortIcon colKey="price" /></div>
                                    <div className="w-[20%] text-right cursor-pointer hover:text-slate-600 dark:hover:text-slate-300" onClick={() => handleSort('change')}>Chg <SortIcon colKey="change" /></div>
                                    <div className="w-[20%] text-right cursor-pointer hover:text-slate-600 dark:hover:text-slate-300" onClick={() => handleSort('percent')}>% <SortIcon colKey="percent" /></div>
                                </div>
                            )}
                        </div>

                        {/* Content Area */}
                        <div className="flex-1 overflow-y-auto">
                            {activeTab === 'alerts' ? (
                                groupedAlerts.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center h-full text-slate-400">
                                        <BellRing size={24} className="mb-2 opacity-50" />
                                        <p className="text-xs">No active alerts</p>
                                        <button onClick={() => setAlertModalOpen(true)} className="mt-2 text-indigo-600 dark:text-indigo-400 text-xs font-bold hover:underline">Create one</button>
                                    </div>
                                ) : (
                                    <div className="p-3 space-y-3">
                                        {groupedAlerts
                                            .filter(({ symbol }) => symbol.toLowerCase().includes(searchQuery.toLowerCase()))
                                            .map(({ symbol, alerts }) => (
                                                <div key={symbol} className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden bg-white dark:bg-slate-900 shadow-sm">
                                                    <div className="bg-slate-50 dark:bg-slate-800 px-3 py-2 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                                                        <div className="flex items-center gap-2">
                                                            <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white ${symbol === 'BTC' ? 'bg-orange-500' : symbol === 'ETH' ? 'bg-blue-600' : symbol === 'SOL' ? 'bg-purple-600' : 'bg-slate-700'}`}>
                                                                {symbol[0]}
                                                            </div>
                                                            <span className="font-bold text-xs text-slate-700 dark:text-slate-300">{symbol}</span>
                                                        </div>
                                                        <span className="text-[10px] bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 px-1.5 py-0.5 rounded-full font-bold">{alerts.length}</span>
                                                    </div>
                                                    <div>
                                                        {alerts.map(alert => (
                                                            <div key={alert.id} className="flex justify-between items-center p-3 border-b border-slate-100 dark:border-slate-800 last:border-0 hover:bg-slate-50 dark:hover:bg-slate-800 group">
                                                                <div>
                                                                    <div className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1">
                                                                        {alert.condition === 'above' ? <ArrowUp size={12} className="text-green-500" /> : <ArrowDown size={12} className="text-red-500" />}
                                                                        ${alert.targetPrice.toLocaleString()}
                                                                    </div>
                                                                    <div className="text-[9px] text-slate-400">Created just now</div>
                                                                </div>
                                                                <button onClick={() => deleteAlert(alert.id)} className="text-slate-300 dark:text-slate-600 hover:text-red-500 p-1 opacity-0 group-hover:opacity-100 transition-all"><Trash2 size={14} /></button>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            ))}
                                    </div>
                                )
                            ) : (
                                processedCoins.length === 0 ? (
                                    <div className="flex flex-col items-center justify-center h-full text-slate-400">
                                        <Search size={24} className="mb-2 opacity-50" />
                                        <p className="text-xs">No coins found</p>
                                    </div>
                                ) : (
                                    processedCoins.map(coin => (
                                        <div
                                            key={coin.symbol}
                                            onClick={() => { setSelectedCoinSymbol(coin.symbol); setForecastResult(null); }}
                                            className={`flex items-center justify-between p-3 cursor-pointer border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors group ${selectedCoinSymbol === coin.symbol ? 'bg-indigo-50 dark:bg-slate-800/50 border-l-2 border-l-indigo-600 dark:border-l-indigo-400' : 'border-l-2 border-l-transparent'}`}
                                        >
                                            <div className="flex items-center gap-2 w-[30%]">
                                                <button onClick={(e) => toggleFavorite(e, coin.symbol)} className="text-slate-300 dark:text-slate-600 hover:text-amber-400 transition-colors">
                                                    <Star size={14} fill={favorites.has(coin.symbol) ? "#fbbf24" : "none"} className={favorites.has(coin.symbol) ? "text-amber-400" : ""} />
                                                </button>
                                                <div>
                                                    <div className="font-bold text-sm text-slate-900 dark:text-slate-100 leading-none">{coin.symbol}</div>
                                                    <div className="text-[9px] text-slate-400">{coin.name}</div>
                                                </div>
                                            </div>
                                            <div className="w-[25%] text-right text-sm font-medium text-slate-700 dark:text-slate-300">${coin.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                                            <div className={`w-[20%] text-right text-xs ${coin.change24h >= 0 ? 'text-green-600 dark:text-green-500' : 'text-red-600 dark:text-red-500'}`}>{coin.change24h > 0 ? '+' : ''}{Math.abs(coin.change24h).toFixed(2)}</div>
                                            <div className={`w-[20%] text-right`}>
                                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${coin.change24h >= 0 ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'}`}>
                                                    {((coin.change24h / (coin.price - coin.change24h)) * 100).toFixed(2)}%
                                                </span>
                                            </div>
                                        </div>
                                    ))
                                )
                            )}
                        </div>
                    </div>

                    {/* Resize Handle */}
                    <div
                        onMouseDown={startResizing}
                        className="h-3 bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 cursor-row-resize flex items-center justify-center hover:bg-indigo-50 dark:hover:bg-slate-800 transition-colors group z-10"
                    >
                        <GripHorizontal size={14} className="text-slate-300 dark:text-slate-600 group-hover:text-indigo-400" />
                    </div>

                    {/* SECTION 2: AI Intelligence */}
                    <div className="flex-1 flex flex-col overflow-hidden bg-slate-50 dark:bg-slate-950">
                        <div className="p-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-white dark:bg-slate-900">
                            <h3 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase flex items-center gap-2">
                                <Zap size={14} /> MarketLens Intelligence
                            </h3>
                            {forecastResult && (
                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${forecastResult.trend === 'Bullish' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'}`}>
                                    {forecastResult.trend}
                                </span>
                            )}
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
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
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-xs text-slate-500 dark:text-slate-400">Confidence</span>
                                            <span className="text-xs font-bold text-slate-900 dark:text-white">{forecastResult.confidenceScore}%</span>
                                        </div>
                                        <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                            <div className={`h-full rounded-full ${forecastResult.confidenceScore > 70 ? 'bg-green-500' : 'bg-yellow-500'}`} style={{ width: `${forecastResult.confidenceScore}%` }}></div>
                                        </div>
                                    </div>

                                    {/* Reasoning */}
                                    <div>
                                        <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 mb-2 uppercase">Analysis</h4>
                                        <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                                            {forecastResult.reasoning}
                                        </p>
                                    </div>

                                    {/* Sources */}
                                    {forecastResult.sources && forecastResult.sources.length > 0 && (
                                        <div>
                                            <h4 className="text-[10px] font-bold text-slate-500 dark:text-slate-400 mb-2 uppercase flex items-center gap-1">
                                                <Globe size={10} /> Grounding
                                            </h4>
                                            <div className="space-y-1">
                                                {forecastResult.sources.slice(0, 2).map((s, i) => (
                                                    <a key={i} href={s.url} target="_blank" className="block text-[10px] text-indigo-600 dark:text-indigo-400 hover:underline truncate">
                                                        {s.title}
                                                    </a>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    <button
                                        onClick={handleForecast}
                                        className="w-full py-1.5 mt-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 rounded text-xs transition flex items-center justify-center gap-2"
                                    >
                                        <RefreshCw size={12} /> Refresh Analysis
                                    </button>
                                </div>
                            ) : loadingForecast ? (
                                // Skeleton Loading State
                                <div className="space-y-4 animate-pulse">
                                    <div className="h-32 bg-slate-100 dark:bg-slate-800 rounded-xl"></div>
                                    <div className="space-y-2">
                                        <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded w-3/4"></div>
                                        <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded w-1/2"></div>
                                    </div>
                                    <div className="h-20 bg-slate-100 dark:bg-slate-800 rounded-lg"></div>
                                </div>
                            ) : (
                                // No forecast - show generate button
                                <div className="flex flex-col items-center justify-center py-12">
                                    <BrainCircuit size={48} className="text-slate-300 dark:text-slate-700 mb-4" />
                                    <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">No forecast generated yet</p>
                                    <button
                                        onClick={handleForecast}
                                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                                    >
                                        <Sparkles size={16} />
                                        Generate AI Forecast
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* --- MODALS --- */}

            {/* Alert Modal */}
            {alertModalOpen && (
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
            )}

            {/* Historical News Modal */}
            {historyModalOpen && (
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
            )}

            {/* Detail Article Modal for History */}
            {selectedHistoryArticle && (
                <ArticleDetailModal article={selectedHistoryArticle} onClose={() => setSelectedHistoryArticle(null)} />
            )}

        </div>
    );
};

export default Dashboard;