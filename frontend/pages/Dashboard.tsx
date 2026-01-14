
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { CoinData, ForecastResult, HistoryPoint, NewsArticle } from '../types';
import LightweightChart from '../components/LightweightChart';

import { Search, ArrowUp, ArrowDown, ChevronLeft, ChevronRight, GripHorizontal, Globe, BrainCircuit, Sparkles, RefreshCw, Zap, Target, ShieldAlert, Check, Loader2, X, SlidersHorizontal, Activity, BarChart2, Star, ChevronDown, List, Hand, Calendar } from 'lucide-react';
import { getTopCoins, getHistoricalData, generateMarketForecast, createSocketConnection, fetchLatestNews, createChatSession } from '../services/apiService';

// Range Configuration for CryptoCompare
// Limit: Number of points. Aggregate: steps to combine. Type: minute/hour/day.
const getRangeParams = (range: string) => {
    switch (range) {
        // Minutes (aggregate from 1-min candles)
        case '5m': return { limit: 60, aggregate: 5, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 5 hours of 5-min candles
        case '15m': return { limit: 96, aggregate: 15, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 24 hours of 15-min candles
        case '30m': return { limit: 48, aggregate: 30, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 24 hours of 30-min candles

        // Hours
        case '1h': return { limit: 60, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 60 x 1min = 1 hour
        case '4h': return { limit: 240, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 240 x 1min = 4 hours

        // Days
        case '1d': return { limit: 24, aggregate: 1, type: 'hour' as const, format: { hour: '2-digit', minute: '2-digit' } as const }; // 24 hours
        case '1w': return { limit: 168, aggregate: 1, type: 'hour' as const, format: { month: 'short', day: 'numeric', hour: '2-digit' } as const }; // 7 days

        // Months
        case '1m': return { limit: 30, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 30 days
        case '3m': return { limit: 90, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 90 days

        // Years
        case '1y': return { limit: 365, aggregate: 1, type: 'day' as const, format: { month: 'short', year: 'numeric' } as const };
        case '5y': return { limit: 1825, aggregate: 1, type: 'day' as const, format: { month: 'short', year: 'numeric' } as const };

        // All
        case 'All': return { limit: 2000, aggregate: 1, type: 'day' as const, format: { year: 'numeric' } as const };

        default: return { limit: 24, aggregate: 1, type: 'hour' as const, format: { hour: '2-digit', minute: '2-digit' } as const };
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
    const [timeRange, setTimeRange] = useState('1d');
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

    // News State
    const [contextNews, setContextNews] = useState<NewsArticle[]>([]);
    const [loadingNews, setLoadingNews] = useState(false);

    const [chartType, setChartType] = useState<'area' | 'candle'>('candle');
    const [indicators, setIndicators] = useState({ rsi: false, macd: false, bollinger: false });
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const indicatorMenuRef = useRef<HTMLDivElement>(null);

    // Date Picker State
    const [showDatePicker, setShowDatePicker] = useState(false);
    const [datePickerMode, setDatePickerMode] = useState<'single' | 'range'>('single');
    const datePickerRef = useRef<HTMLDivElement>(null);
    const [tempSingleDate, setTempSingleDate] = useState('');

    // Close menus when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (indicatorMenuRef.current && !indicatorMenuRef.current.contains(event.target as Node)) {
                setShowIndicatorMenu(false);
            }
            if (datePickerRef.current && !datePickerRef.current.contains(event.target as Node)) {
                setShowDatePicker(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);





    // WebSocket connection status and visual feedback
    const [wsStatus, setWsStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
    const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null);
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

            setWsStatus('connected');
            // Request to join room for kline and trade updates
            socket.emit('join-room', { symbol, type: 'kline' });
            socket.emit('join-room', { symbol, type: 'trade' });
        });

        socket.on('disconnect', () => {

            setWsStatus('disconnected');
        });

        socket.on('reconnect', () => {

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
                const kline = payload.data;


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

    // Fetch news when chart range or selected coin changes
    useEffect(() => {
        const fetchContextNews = async () => {
            if (!chartVisibleRange) return;

            setLoadingNews(true);
            try {
                // Convert unix timestamp to ISO string
                const startIso = new Date(chartVisibleRange.from * 1000).toISOString();
                const endIso = new Date(chartVisibleRange.to * 1000).toISOString();

                // Fetch news filtered by date AND selected coin tag
                const articles = await fetchLatestNews(startIso, endIso, selectedCoin.symbol);
                setContextNews(articles);
            } catch (err) {
                console.error("Failed to fetch context news", err);
            } finally {
                setLoadingNews(false);
            }
        };

        // Debounce slightly to avoid too many requests during drag
        const timeoutId = setTimeout(fetchContextNews, 500);
        return () => clearTimeout(timeoutId);
    }, [chartVisibleRange, selectedCoin?.symbol]);

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
                setShowDatePicker(false);
            }
        }
    };

    const handleSingleDateApply = () => {
        if (tempSingleDate) {
            // Create start and end of that day in UTC
            const start = Math.floor(new Date(tempSingleDate + 'T00:00:00Z').getTime() / 1000);
            const end = Math.floor(new Date(tempSingleDate + 'T23:59:59Z').getTime() / 1000);

            setChartVisibleRange({ from: start, to: end });
            setHasActiveRange(true);
            setTimeRange('Custom');
            setShowDatePicker(false);
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
                            {/* Time Range Buttons - Simplified */}
                            <div className="flex flex-wrap gap-1">
                                {['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1m', '3m', '1y', '5y', 'All'].map((range) => (
                                    <button
                                        key={range}
                                        onClick={() => setTimeRange(range)}
                                        className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${timeRange === range
                                            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20'
                                            : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 dark:text-slate-400'
                                            }`}
                                    >    {range === '1m' ? '1M' : range === '3m' ? '3M' : range}
                                    </button>
                                ))}
                            </div>

                            {/* Custom Range Inputs & Hand Tool */}
                            <div className="flex items-center gap-2">

                                {/* New Modern Date Picker */}
                                <div className="relative" ref={datePickerRef}>
                                    <button
                                        onClick={() => setShowDatePicker(!showDatePicker)}
                                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg border transition-all ${showDatePicker || hasActiveRange
                                            ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800'
                                            : 'bg-white dark:bg-slate-900 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800'
                                            }`}
                                    >
                                        <Calendar size={14} />
                                        {hasActiveRange ? "Custom Range" : "Select Date"}
                                        <ChevronDown size={12} className={`transition-transform ${showDatePicker ? 'rotate-180' : ''}`} />
                                    </button>

                                    {/* Date Picker Popover */}
                                    {showDatePicker && (
                                        <div className="absolute right-0 top-full mt-2 w-72 bg-white dark:bg-slate-900 rounded-xl shadow-xl border border-slate-200 dark:border-slate-800 z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200 p-3">
                                            {/* Tabs */}
                                            <div className="flex p-1 bg-slate-100 dark:bg-slate-800 rounded-lg mb-4">
                                                <button
                                                    onClick={() => setDatePickerMode('single')}
                                                    className={`flex-1 text-xs font-bold py-1.5 rounded-md transition-all ${datePickerMode === 'single' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-500 dark:text-slate-400'}`}
                                                >
                                                    One Day
                                                </button>
                                                <button
                                                    onClick={() => setDatePickerMode('range')}
                                                    className={`flex-1 text-xs font-bold py-1.5 rounded-md transition-all ${datePickerMode === 'range' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-500 dark:text-slate-400'}`}
                                                >
                                                    Range
                                                </button>
                                            </div>

                                            {/* Content */}
                                            <div className="space-y-3">
                                                {datePickerMode === 'single' ? (
                                                    <div className="space-y-3">
                                                        <div>
                                                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Select Date</label>
                                                            <input
                                                                type="date"
                                                                value={tempSingleDate}
                                                                onChange={(e) => setTempSingleDate(e.target.value)}
                                                                className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                            />
                                                        </div>
                                                        <button
                                                            onClick={handleSingleDateApply}
                                                            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2 rounded-lg transition-colors"
                                                        >
                                                            Apply Single Day
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <div className="space-y-3">
                                                        <div className="grid grid-cols-2 gap-2">
                                                            <div>
                                                                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">From</label>
                                                                <input
                                                                    type="datetime-local"
                                                                    value={manualDateFrom}
                                                                    onChange={(e) => setManualDateFrom(e.target.value)}
                                                                    className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-2 text-[10px] text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                                />
                                                            </div>
                                                            <div>
                                                                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">To</label>
                                                                <input
                                                                    type="datetime-local"
                                                                    value={manualDateTo}
                                                                    onChange={(e) => setManualDateTo(e.target.value)}
                                                                    className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-2 text-[10px] text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                                />
                                                            </div>
                                                        </div>
                                                        <button
                                                            onClick={handleManualRangeApply}
                                                            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2 rounded-lg transition-colors"
                                                        >
                                                            Apply Range
                                                        </button>
                                                    </div>
                                                )}

                                                {/* Clear Selection */}
                                                {hasActiveRange && (
                                                    <button
                                                        onClick={() => { clearRangeSelection(); setShowDatePicker(false); }}
                                                        className="w-full text-center text-xs text-rose-500 hover:text-rose-600 font-medium mt-1"
                                                    >
                                                        Clear Selection
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    )}
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
                        className="w-full md:w-80 lg:w-96 flex flex-col bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 h-[calc(100vh-3.5rem)] overflow-hidden transition-all duration-300"
                    >
                        {/* Header */}
                        <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex-none z-10">
                            <h3 className="text-sm font-bold text-slate-800 dark:text-white uppercase flex items-center gap-2">
                                <Zap size={16} className="text-indigo-500" />
                                Market Intelligence
                            </h3>
                            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                                AI-powered analysis & news for {selectedCoin.name}
                            </p>
                        </div>

                        {/* Sidebar Content: Split View */}
                        <div className="flex-1 flex flex-col overflow-hidden min-h-0">

                            {/* TOP: News Context (50%) */}
                            <div className="flex-1 flex flex-col border-b border-slate-200 dark:border-slate-800 min-h-0 overflow-hidden">
                                <div className="p-3 bg-slate-50 dark:bg-slate-950/50 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center flex-none">
                                    <h4 className="font-bold text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                        <Globe size={14} className="text-blue-500" />
                                        Contextual News
                                    </h4>
                                    {hasActiveRange && (
                                        <span className="text-[10px] font-medium text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20 px-2 py-0.5 rounded-full">
                                            Filtered by Range
                                        </span>
                                    )}
                                </div>

                                <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar">
                                    {loadingNews ? (
                                        <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-2 py-8">
                                            <Loader2 size={20} className="animate-spin" />
                                            <p className="text-xs">Finding insights...</p>
                                        </div>
                                    ) : contextNews.length > 0 ? (
                                        contextNews.map((article) => (
                                            <div key={article.id} className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-100 dark:border-slate-700 hover:border-indigo-200 dark:hover:border-indigo-800 transition-all group shadow-sm">
                                                <div className="flex justify-between items-start mb-1.5">
                                                    <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20 px-2 py-0.5 rounded-full">
                                                        {article.source}
                                                    </span>
                                                    <span className="text-[10px] text-slate-400">
                                                        {new Date(article.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                                    </span>
                                                </div>
                                                <a href={article.url} target="_blank" rel="noopener noreferrer" className="block">
                                                    <h5 className="font-bold text-xs text-slate-800 dark:text-slate-200 mb-1.5 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors line-clamp-2">
                                                        {article.title}
                                                    </h5>
                                                </a>
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${article.sentiment === 'Positive' ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-900/20 dark:text-emerald-400' :
                                                        article.sentiment === 'Negative' ? 'bg-rose-50 text-rose-600 dark:bg-rose-900/20 dark:text-rose-400' :
                                                            'bg-slate-50 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                                                        }`}>
                                                        {article.sentiment}
                                                    </span>
                                                    {article.tag && (
                                                        <span className="text-[10px] text-slate-400 border border-slate-100 dark:border-slate-700 px-1.5 py-0.5 rounded-full">
                                                            {article.tag}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        ))
                                    ) : (
                                        <div className="flex flex-col items-center justify-center h-full text-slate-400 py-8">
                                            <Globe size={24} className="mb-2 opacity-20" />
                                            <p className="text-xs text-center">No news related to {selectedCoin.symbol} in this range.</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* BOTTOM: AI Forecast & Summary (50%) */}
                            <div className="flex-1 flex flex-col bg-slate-50 dark:bg-slate-950 min-h-0 overflow-hidden">
                                <div className="p-3 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-white dark:bg-slate-900 flex-none">
                                    <h4 className="font-bold text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                        <BrainCircuit size={14} className="text-purple-500" />
                                        AI Forecast
                                    </h4>
                                </div>
                                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                                    {forecastResult ? (
                                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-500">

                                            {/* Recommendation Card */}
                                            {forecastResult.recommendation && (
                                                <div className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                                                    <div className="flex justify-between items-center mb-3">
                                                        <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide flex items-center gap-1">
                                                            <Target size={14} /> Trade Signal
                                                        </h4>
                                                        <span className={`px-2 py-1 rounded-lg text-xs font-bold ${forecastResult.recommendation.action === 'Buy' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' :
                                                            forecastResult.recommendation.action === 'Sell' ? 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400' :
                                                                'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-400'
                                                            }`}>
                                                            {forecastResult.recommendation.action}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-4">
                                                        <div className="flex-1">
                                                            <div className="text-xs text-slate-500 mb-1">Confidence</div>
                                                            <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                                                <div
                                                                    className={`h-full rounded-full ${forecastResult.confidenceScore > 75 ? 'bg-emerald-500' :
                                                                        forecastResult.confidenceScore > 50 ? 'bg-amber-500' : 'bg-rose-500'
                                                                        }`}
                                                                    style={{ width: `${forecastResult.confidenceScore}%` }}
                                                                ></div>
                                                            </div>
                                                            <div className="text-right text-[10px] font-bold mt-1 text-slate-600 dark:text-slate-400">
                                                                {forecastResult.confidenceScore}%
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Market Summary */}
                                            {forecastResult.marketSummary && (
                                                <div className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                                                    <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-1">
                                                        <Activity size={14} /> Analysis
                                                    </h4>
                                                    <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                                                        {forecastResult.marketSummary}
                                                    </p>
                                                </div>
                                            )}

                                            {/* Reasoning */}
                                            {forecastResult.reasoning && (
                                                <div className="bg-indigo-50 dark:bg-indigo-900/10 p-4 rounded-xl border border-indigo-100 dark:border-indigo-900/30">
                                                    <h4 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wide mb-2 flex items-center gap-1">
                                                        <Sparkles size={14} /> AI Reasoning
                                                    </h4>
                                                    <p className="text-xs italic text-indigo-800 dark:text-indigo-200">
                                                        "{forecastResult.reasoning}"
                                                    </p>
                                                </div>
                                            )}

                                            <div className="pt-2">
                                                <button
                                                    onClick={handleForecast}
                                                    className="w-full py-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2"
                                                >
                                                    <RefreshCw size={12} /> Regenerate Analysis
                                                </button>
                                            </div>
                                        </div>
                                    ) : loadingForecast ? (
                                        <div className="space-y-4 animate-pulse p-2">
                                            <div className="h-32 bg-slate-200 dark:bg-slate-800 rounded-xl"></div>
                                            <div className="h-24 bg-slate-200 dark:bg-slate-800 rounded-xl"></div>
                                            <div className="space-y-2">
                                                <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded w-3/4"></div>
                                                <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded w-1/2"></div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center justify-center h-full py-10 px-4 text-center text-slate-400">
                                            <BrainCircuit size={32} className="mb-3 opacity-20" />
                                            <h4 className="text-sm font-bold text-slate-500 dark:text-slate-400 mb-1">Ready to Analyze</h4>
                                            <p className="text-xs mb-4">Generate AI-powered prediction for {selectedCoin.symbol}.</p>
                                            <button
                                                onClick={handleForecast}
                                                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-bold transition-all shadow-md shadow-indigo-500/20 flex items-center gap-2"
                                            >
                                                <Sparkles size={14} />
                                                Generate Forecast
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>

                        </div>
                    </div>
                )}

            {/* -- MODALS -- */}

            {/* Alert Modal */}

        </div >
    );
};

// Chat Interface Component
const ChatInterface: React.FC = () => {
    const [messages, setMessages] = useState<{ role: 'user' | 'model'; content: string }[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const sessionRef = useRef<any>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initialize chat session
        sessionRef.current = createChatSession();

        // Add initial greeting
        setMessages([{ role: 'model', content: "Hello! I'm Sibyl. Ask me anything about this coin's price action, news, or technicals." }]);
    }, []);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || loading || !sessionRef.current) return;

        const userMsg = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setLoading(true);

        try {
            const result = await sessionRef.current.sendMessage(userMsg);
            setMessages(prev => [...prev, { role: 'model', content: result.response.text() }]);
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { role: 'model', content: "Sorry, I encountered an error. Please try again." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full overflow-hidden">
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm ${msg.role === 'user'
                            ? 'bg-indigo-600 text-white rounded-br-none'
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-bl-none border border-slate-200 dark:border-slate-700'
                            }`}>
                            {msg.content}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-bl-none px-4 py-3 border border-slate-200 dark:border-slate-700">
                            <Loader2 size={16} className="animate-spin text-slate-400" />
                        </div>
                    </div>
                )}
            </div>

            <div className="p-3 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
                <div className="relative flex items-center">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask Sibyl..."
                        className="w-full pl-4 pr-10 py-2.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 text-sm"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || loading}
                        className="absolute right-2 p-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white rounded-lg transition-colors"
                    >
                        <Sparkles size={14} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
