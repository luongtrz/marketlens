
import React, { useState, useEffect, useLayoutEffect, useRef, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import { CoinData, ForecastResult, HistoryPoint, NewsArticle } from '../types';
import LightweightChart from '../components/LightweightChart';

import { Search, ArrowUp, ArrowDown, ChevronLeft, ChevronRight, GripHorizontal, Globe, BrainCircuit, Sparkles, RefreshCw, Zap, Target, ShieldAlert, Check, Loader2, X, SlidersHorizontal, Activity, BarChart2, Star, ChevronDown, List, Hand, Calendar } from 'lucide-react';
import { getTopCoins, getHistoricalData, generateMarketForecast, fetchLatestNews, getCachedLatestNews, getCachedTopCoins, createChatSession, MarketWebSocket } from '../services/apiService';
import { useAuth } from '../context/AuthContext';
import { formatDateToLocalWithOffset } from '../utils/formatters';

// Range Configuration for CryptoCompare
// Limit: Number of points. Aggregate: steps to combine. Type: minute/hour/day.
// Binance supported intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
const getRangeParams = (range: string) => {
    switch (range) {
        // Minutes (aggregate from 1-min candles)
        case '1m': return { limit: 120, aggregate: 1, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 2 hours
        case '3m': return { limit: 80, aggregate: 3, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 4 hours
        case '5m': return { limit: 72, aggregate: 5, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 6 hours
        case '15m': return { limit: 96, aggregate: 15, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 24 hours
        case '30m': return { limit: 96, aggregate: 30, type: 'minute' as const, format: { hour: '2-digit', minute: '2-digit', hour12: false } as const }; // 48 hours

        // Hours (aggregate from 1-hour candles)
        case '1h': return { limit: 168, aggregate: 1, type: 'hour' as const, format: { month: 'short', day: 'numeric', hour: '2-digit' } as const }; // 7 days
        case '2h': return { limit: 168, aggregate: 2, type: 'hour' as const, format: { month: 'short', day: 'numeric', hour: '2-digit' } as const }; // 14 days
        case '4h': return { limit: 180, aggregate: 4, type: 'hour' as const, format: { month: 'short', day: 'numeric', hour: '2-digit' } as const }; // 30 days
        case '6h': return { limit: 120, aggregate: 6, type: 'hour' as const, format: { month: 'short', day: 'numeric' } as const }; // 30 days
        case '8h': return { limit: 90, aggregate: 8, type: 'hour' as const, format: { month: 'short', day: 'numeric' } as const }; // 30 days
        case '12h': return { limit: 60, aggregate: 12, type: 'hour' as const, format: { month: 'short', day: 'numeric' } as const }; // 30 days

        // Days (aggregate from 1-day candles)
        case '1d': return { limit: 90, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 90 days
        case '3d': return { limit: 90, aggregate: 3, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 270 days

        // Weeks (aggregate from daily)
        case '1w': return { limit: 52, aggregate: 7, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 1 year

        // Months (Binance native: 1M only)
        case '1M': return { limit: 30, aggregate: 1, type: 'day' as const, format: { month: 'short', day: 'numeric' } as const }; // 30 days = 1 month

        default: return { limit: 24, aggregate: 1, type: 'hour' as const, format: { hour: '2-digit', minute: '2-digit' } as const };
    }
};

type SortKey = 'price' | 'change' | 'percent';
type SortDirection = 'asc' | 'desc';

const CONTEXT_NEWS_PAGE_SIZE = 8;

const makeShellCoin = (symbol: string): CoinData => ({
    symbol,
    name: symbol,
    price: 0,
    change24h: 0,
    volume: '-',
    marketCap: '-',
    history: [],
});

const getInitialCoins = (): CoinData[] => {
    const cached = getCachedTopCoins();
    return cached?.length ? cached : [makeShellCoin('BTC'), makeShellCoin('ETH')];
};

const percentChange = (price: number, change: number): number => {
    const previous = price - change;
    if (!Number.isFinite(previous) || previous === 0) return 0;
    return (change / previous) * 100;
};

const CoinIcon: React.FC<{ symbol: string; size?: 'sm' | 'md' }> = ({ symbol, size = 'md' }) => {
    const normalized = symbol.toUpperCase();
    const box = size === 'sm' ? 'w-5 h-5 text-[11px]' : 'w-6 h-6 text-xs';

    if (normalized === 'BTC') {
        return (
            <div className={`${box} rounded-full flex items-center justify-center font-bold text-white bg-[#f7931a] shadow-sm`}>
                ₿
            </div>
        );
    }

    if (normalized === 'ETH') {
        return (
            <div className={`${box} rounded-full flex items-center justify-center bg-[#627eea] shadow-sm`}>
                <div className="relative h-4 w-2.5">
                    <div className="absolute left-1/2 top-0 h-0 w-0 -translate-x-1/2 border-l-[5px] border-r-[5px] border-b-[8px] border-l-transparent border-r-transparent border-b-white/95" />
                    <div className="absolute left-1/2 bottom-0 h-0 w-0 -translate-x-1/2 border-l-[5px] border-r-[5px] border-t-[8px] border-l-transparent border-r-transparent border-t-white/80" />
                </div>
            </div>
        );
    }

    return (
        <div className={`${box} rounded-full flex items-center justify-center font-bold text-white shadow-sm ${normalized === 'SOL' ? 'bg-purple-600' : 'bg-slate-700'}`}>
            {normalized[0]}
        </div>
    );
};

const Dashboard: React.FC = () => {
    const { isAuthenticated } = useAuth();
    const getNewsCardAccent = (sentiment?: string) => {
        if (sentiment === 'Positive') return 'border-emerald-200 ring-emerald-100';
        if (sentiment === 'Negative') return 'border-red-200 ring-red-100';
        return 'border-amber-200 ring-amber-100';
    };

    const getNewsCardHover = (sentiment?: string) => {
        if (sentiment === 'Positive') return 'hover:border-emerald-300 hover:ring-emerald-200/70 hover:shadow-emerald-200/30';
        if (sentiment === 'Negative') return 'hover:border-red-300 hover:ring-red-200/70 hover:shadow-red-200/30';
        return 'hover:border-amber-300 hover:ring-amber-200/70 hover:shadow-amber-200/30';
    };
    const [coins, setCoins] = useState<CoinData[]>(getInitialCoins);
    const [selectedCoinSymbol, setSelectedCoinSymbol] = useState<string>(() => {
        return localStorage.getItem('marketlens_selected_coin') || 'BTC';
    });
    const lastContextNewsFetchKeyRef = useRef<string>('');

    useEffect(() => {
        localStorage.setItem('marketlens_selected_coin', selectedCoinSymbol);
    }, [selectedCoinSymbol]);

    /** New coin ⇒ drop manual range before paint so news fetch never uses the previous coin's range. */
    useLayoutEffect(() => {
        setChartVisibleRange(null);
        setHasActiveRange(false);
        lastContextNewsFetchKeyRef.current = '';
    }, [selectedCoinSymbol]);
    const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);
    const [loadingForecast, setLoadingForecast] = useState(false);
    const [combinedChartData, setCombinedChartData] = useState<any[]>([]);
    const [timeRange, setTimeRange] = useState('1d');
    const [loadingMarket, setLoadingMarket] = useState(false);
    const [loadingChart, setLoadingChart] = useState(true);
    const [loadingMoreHistory, setLoadingMoreHistory] = useState(false);

    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isCoinDropdownOpen, setIsCoinDropdownOpen] = useState(false);

    // Persist WS connection
    const wsRef = useRef<MarketWebSocket | null>(null);
    const currentSymbolRef = useRef<string>('');
    const sidebarRef = useRef<HTMLDivElement>(null);
    const contextNewsListScrollRef = useRef<HTMLDivElement>(null);
    const lastChartFetchKeyRef = useRef<string>('');

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
    const [contextNews, setContextNews] = useState<NewsArticle[]>(() => {
        return getCachedLatestNews(undefined, undefined, localStorage.getItem('marketlens_selected_coin') || 'BTC') || [];
    });
    const [contextNewsPage, setContextNewsPage] = useState(1);
    const [loadingNews, setLoadingNews] = useState(false);

    const [chartType, setChartType] = useState<'area' | 'candle'>('candle');
    const [indicators, setIndicators] = useState({ rsi: false, macd: false, bollinger: false });
    const [showIndicatorMenu, setShowIndicatorMenu] = useState(false);
    const [showForecast, setShowForecast] = useState(true);
    const indicatorMenuRef = useRef<HTMLDivElement>(null);

    // Date Picker State
    const [showDatePicker, setShowDatePicker] = useState(false);
    const [datePickerMode, setDatePickerMode] = useState<'single' | 'range'>('single');
    const [showModeSelector, setShowModeSelector] = useState(false);
    const datePickerRef = useRef<HTMLDivElement>(null);
    const [tempSingleDate, setTempSingleDate] = useState('');

    // Hover State for Chart OHLCV
    const [hoveredCandle, setHoveredCandle] = useState<{ open?: number, high?: number, low?: number, close?: number, volume?: number, time: number | null } | null>(null);

    // Close menus when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (indicatorMenuRef.current && !indicatorMenuRef.current.contains(event.target as Node)) {
                setShowIndicatorMenu(false);
            }
            if (datePickerRef.current && !datePickerRef.current.contains(event.target as Node)) {
                setShowDatePicker(false);
                setShowModeSelector(false);
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
    const chartResetKey = `${selectedCoinSymbol}:${timeRange}`;

    useEffect(() => {
        const fetchMarket = async () => {
            if (coins.length === 0) {
                setLoadingMarket(true);
            }
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
        setCombinedChartData([]);
        setHoveredCandle(null);
        setLoadingChart(true);
        lastChartFetchKeyRef.current = '';
    }, [chartResetKey]);

    useEffect(() => {
        let cancelled = false;

        const fetchChart = async () => {
            if (!selectedCoin) {
                setLoadingChart(false);
                return;
            }
            // Skip fetch when Custom range is selected - Custom is for news filtering only, not chart data
            if (timeRange === 'Custom') {
                setLoadingChart(false);
                return;
            }

            setLoadingChart(true);
            const { limit, aggregate, type, format } = getRangeParams(timeRange);
            const fetchKey = `${selectedCoin.symbol}:${timeRange}:${limit}:${aggregate}:${type}:${forecastResult ? 'forecast' : 'plain'}`;
            if (fetchKey === lastChartFetchKeyRef.current && combinedChartData.length > 0) {
                setLoadingChart(false);
                return;
            }
            // CryptoCompare uses Symbol directly
            try {
                const history = await getHistoricalData(selectedCoin.symbol, limit, aggregate, type);
                if (cancelled) return;

                if (history.length === 0) {
                    setCombinedChartData((prev) => prev);
                    return;
                }

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
                lastChartFetchKeyRef.current = fetchKey;
            } finally {
                if (!cancelled) {
                    setLoadingChart(false);
                }
            }
        };

        fetchChart();
        return () => {
            cancelled = true;
        };
    }, [selectedCoin?.symbol, forecastResult, timeRange]);

    const handleLoadMore = async (oldestTimestamp: number) => {
        if (!selectedCoin || loadingMoreHistory || timeRange === 'Custom') return;
        
        setLoadingMoreHistory(true);
        try {
            const { limit, aggregate, type, format } = getRangeParams(timeRange);
            // Fetch older data with endTime = oldestTimestamp
            const olderHistory = await getHistoricalData(selectedCoin.symbol, limit, aggregate, type, oldestTimestamp);
            
            if (olderHistory.length > 0) {
                // Filter out the exact same timestamp if it overlaps
                const filteredHistory = olderHistory.filter(h => h.ts < oldestTimestamp);
                
                const formattedOlderData = filteredHistory.map(h => ({
                    ...h,
                    time: new Date(h.ts).toLocaleString('en-US', format)
                }));

                setCombinedChartData(prevData => {
                    // Prepend older data to existing data
                    return [...formattedOlderData, ...prevData];
                });
            }
        } catch (error) {
            console.error('Failed to load more history:', error);
        } finally {
            setLoadingMoreHistory(false);
        }
    };


    // Keep refs up-to-date for WebSocket callbacks
    useEffect(() => {
        if (selectedCoin) {
            currentSymbolRef.current = selectedCoin.symbol;
            lastPriceRef.current = selectedCoin.price;
        }
    }, [selectedCoin]);

    // 1. Initialize WebSocket exactly once
    useEffect(() => {
        // Robust Throttle Mechanism (Leading & Trailing)
        const processTradeUpdate = (price: number) => {
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

        wsRef.current = new MarketWebSocket({
            onStatusChange: (status) => {
                if (status === 'connected') {
                    setWsStatus('connected');
                } else {
                    setWsStatus('disconnected');
                }
            },
            onTrade: (tradeSymbol, price) => {
                if (tradeSymbol !== currentSymbolRef.current) return;

                const now = Date.now();
                const { lastRun, timeout } = throttleRef.current;

                if (now - lastRun >= TRADE_THROTTLE_MS) {
                    // Leading edge: Execute immediately
                    processTradeUpdate(price);
                } else {
                    // Trailing edge: Schedule update
                    throttleRef.current.pendingArg = price;

                    if (!timeout) {
                        const delay = TRADE_THROTTLE_MS - (now - lastRun);
                        throttleRef.current.timeout = setTimeout(() => {
                            if (throttleRef.current.pendingArg !== null) {
                                processTradeUpdate(throttleRef.current.pendingArg);
                            }
                        }, delay);
                    }
                }
            },
            onKline: (klineSymbol, klineData) => {
                if (klineSymbol !== currentSymbolRef.current) return;

                setCombinedChartData(prevData => {
                    if (prevData.length === 0) return prevData;

                    const newData = [...prevData];

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

                    // Determine interval size based on historical data points
                    const intervalMs = targetIndex > 0 ? targetPoint.ts - newData[targetIndex - 1].ts : 60000;

                    if (targetPoint && klineData.time >= targetPoint.ts + intervalMs) {
                        // The incoming kline is past the current interval bucket, so push a new candle
                        const nextBucketTs = targetPoint.ts + intervalMs;
                        newData.push({
                            ...targetPoint,
                            ts: nextBucketTs,
                            time: new Date(nextBucketTs).toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true }),
                            price: klineData.close,
                            open: targetPoint.close, // new open is previous close
                            high: klineData.high,
                            low: klineData.low,
                            close: klineData.close,
                            volume: klineData.volume
                        });
                    } else if (targetPoint && klineData.time >= targetPoint.ts) {
                        // Update current bucket with new live prices, but keep the exact original timestamp
                        newData[targetIndex] = {
                            ...targetPoint,
                            price: klineData.close,
                            high: Math.max(targetPoint.high || klineData.high, klineData.high),
                            low: Math.min(targetPoint.low || klineData.low, klineData.low),
                            close: klineData.close,
                            // We don't overwrite open to avoid jumping from original open
                            volume: (targetPoint.volume || 0) + klineData.volume
                        };
                    }

                    return newData;
                });
            }
        });

        wsRef.current.connect();

        return () => {
            if (throttleRef.current.timeout) clearTimeout(throttleRef.current.timeout);
            wsRef.current?.disconnect();
        };
    }, []);

    // 2. Subscribe/Unsubscribe on coin change without disconnecting
    useEffect(() => {
        if (!selectedCoin || !wsRef.current) return;
        
        // Use a short timeout to ensure WebSocket is connected before subscribing,
        // although MarketWebSocket queues subscriptions internally when reconnecting,
        // if it's the first render, it's safer to just call subscribe.
        const newSymbol = selectedCoin.symbol;
        
        wsRef.current.subscribe(newSymbol, 'kline');
        wsRef.current.subscribe(newSymbol, 'trade');

        return () => {
            // Cleanup subscriptions on unmount or coin change
            wsRef.current?.unsubscribe(newSymbol, 'kline');
            wsRef.current?.unsubscribe(newSymbol, 'trade');
        };
    }, [selectedCoin?.symbol]);



    const handleForecast = async () => {
        if (!isAuthenticated) {
            return;
        }
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

    // Headlines for the selected coin; when a chart time range is active, filter to [start, end] (ISO UTC).
    useEffect(() => {
        if (!selectedCoin?.symbol) {
            return;
        }

        const useRangeFilter = Boolean(hasActiveRange && chartVisibleRange);
        const startIso =
            useRangeFilter && chartVisibleRange
                ? new Date(chartVisibleRange.from * 1000).toISOString()
                : undefined;
        const endIso =
            useRangeFilter && chartVisibleRange
                ? new Date(chartVisibleRange.to * 1000).toISOString()
                : undefined;

        const fetchKey =
            `${selectedCoin.symbol}:` +
            (useRangeFilter && chartVisibleRange
                ? `${chartVisibleRange.from}-${chartVisibleRange.to}`
                : 'all');
        if (fetchKey === lastContextNewsFetchKeyRef.current) {
            return;
        }

        let cancelled = false;

        const fetchContextNews = async () => {
            setLoadingNews(true);
            try {
                const articles = await fetchLatestNews(startIso, endIso, selectedCoin.symbol);
                if (!cancelled) {
                    setContextNewsPage(1);
                    setContextNews(articles);
                    lastContextNewsFetchKeyRef.current = fetchKey;
                }
            } catch (err) {
                if (!cancelled) {
                    console.error("Failed to fetch context news", err);
                    setContextNews([]);
                    lastContextNewsFetchKeyRef.current = fetchKey;
                }
            } finally {
                if (!cancelled) {
                    setLoadingNews(false);
                }
            }
        };

        const timeoutId = setTimeout(() => {
            void fetchContextNews();
        }, 120);

        return () => {
            cancelled = true;
            clearTimeout(timeoutId);
        };
    }, [selectedCoin?.symbol, hasActiveRange, chartVisibleRange?.from, chartVisibleRange?.to]);

    const contextNewsTotalPages = Math.max(1, Math.ceil(contextNews.length / CONTEXT_NEWS_PAGE_SIZE));
    const contextNewsSafePage = Math.min(contextNewsPage, contextNewsTotalPages);
    const contextNewsPageArticles = useMemo(() => {
        const start = (contextNewsSafePage - 1) * CONTEXT_NEWS_PAGE_SIZE;
        return contextNews.slice(start, start + CONTEXT_NEWS_PAGE_SIZE);
    }, [contextNews, contextNewsSafePage]);

    useEffect(() => {
        contextNewsListScrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    }, [contextNewsSafePage]);

    useEffect(() => {
        const tp = Math.max(1, Math.ceil(contextNews.length / CONTEXT_NEWS_PAGE_SIZE));
        setContextNewsPage((p) => (p > tp ? tp : p));
    }, [contextNews.length]);

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
                percent: percentChange(selectedCoin.price, selectedCoin.change24h)
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

        // Helper to format timestamp to datetime-local string (local time for input display)
        const formatDate = (ts: number) => {
            const d = new Date(ts * 1000);
            const pad = (n: number) => n.toString().padStart(2, '0');
            // Use local time for input display (not UTC)
            return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
        };

        // One Day mode: Single click = 6h before to 3h after
        if (datePickerMode === 'single') {
            console.log('[One Day Click] Time:', time, '=', new Date(time * 1000).toISOString());

            const twelveHoursS = 12 * 60 * 60;
            const start = time - twelveHoursS;
            const end = time + twelveHoursS;

            setChartVisibleRange({ from: start, to: end });
            setTempSingleDate(formatDate(time));
            setHasActiveRange(true);
            setTimeRange('Custom');
            setIsRangeSelecting(false);
            return;
        }

        // Range mode: Two clicks
        if (rangeStart === null) {
            // First Click: Start Selection
            console.log('[Range Click] Start point:', time, '=', new Date(time * 1000).toISOString());
            setRangeStart(time);
            setManualDateFrom(formatDate(time));
            setManualDateTo(''); // Clear End Date to indicate pending selection
        } else {
            // Second Click: End Selection
            const start = Math.min(rangeStart, time);
            const end = Math.max(rangeStart, time);

            console.log('[Range Click] End point:', time, '=', new Date(time * 1000).toISOString());
            console.log('[Range Click] Final range:', new Date(start * 1000).toISOString(), 'to', new Date(end * 1000).toISOString());

            // Update Visible Range - this triggers news fetch via useEffect
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
            // datetime-local gives local time, don't append 'Z' which treats as UTC
            const start = Math.floor(new Date(manualDateFrom).getTime() / 1000);
            const end = Math.floor(new Date(manualDateTo).getTime() / 1000);

            console.log('[Range Filter] FROM:', manualDateFrom, '-> start:', new Date(start * 1000).toISOString());
            console.log('[Range Filter] TO:', manualDateTo, '-> end:', new Date(end * 1000).toISOString());

            if (start < end) {
                setChartVisibleRange({ from: start, to: end });
                setHasActiveRange(true);
                setTimeRange('Custom');
                setShowDatePicker(false);
            } else {
                console.warn('[Range Filter] Invalid range: start >= end');
            }
        }
    };

    const handleSingleDateApply = () => {
        if (tempSingleDate) {
            // datetime-local gives local time in format: "2026-01-08T12:26"
            // Parse as local time (don't append 'Z' which would treat as UTC)
            const selectedTime = new Date(tempSingleDate).getTime();
            const twelveHoursMs = 12 * 60 * 60 * 1000;

            const start = Math.floor((selectedTime - twelveHoursMs) / 1000);
            const end = Math.floor((selectedTime + twelveHoursMs) / 1000);

            console.log('[Date Filter] Selected:', tempSingleDate);
            console.log('[Date Filter] Range:', new Date(start * 1000).toISOString(), 'to', new Date(end * 1000).toISOString());

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
                    <div className="px-4 py-2">
                        {/* Row 1: Title & Price */}
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                            <div className="relative">
                                <button
                                    onClick={() => setIsCoinDropdownOpen(!isCoinDropdownOpen)}
                                    className="flex items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-800 p-1 -ml-1 rounded-lg transition-colors text-left group"
                                >
                                    <CoinIcon symbol={selectedCoin.symbol} />

                                    <div className="flex items-center gap-3">
                                        <div className="flex items-center gap-1.5">
                                            <h1 className="text-sm font-bold text-slate-900 dark:text-white leading-none flex items-center gap-1">
                                                {selectedCoin.name}
                                                <ChevronDown size={14} className={`text-slate-400 transition-transform ${isCoinDropdownOpen ? 'rotate-180' : ''}`} />
                                            </h1>
                                            <div className={`w-1.5 h-1.5 rounded-full ${wsStatus === 'connected' ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} title={wsStatus === 'connected' ? 'Live' : 'Connecting...'}></div>
                                        </div>
                                        <div className="flex items-baseline gap-2">
                                            <span className={`text-base font-bold text-slate-900 dark:text-white tracking-tight transition-all duration-300 ${priceFlash}`}>
                                                ${headerStats.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                            </span>
                                            <div className={`flex items-center gap-1 font-bold text-[10px] ${headerStats.change >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                {headerStats.change > 0 ? '+' : ''}{headerStats.change.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({Math.abs(headerStats.percent).toFixed(2)}%)
                                            </div>
                                            {hoveredCandle && hoveredCandle.close !== undefined && (
                                                <div className="hidden lg:flex items-center gap-2 ml-2 text-[10px] font-medium text-slate-500 dark:text-slate-400 border-l border-slate-200 dark:border-slate-700 pl-3">
                                                    <span>O: <span className={hoveredCandle.open! > hoveredCandle.close ? 'text-rose-500 dark:text-rose-400' : 'text-emerald-500 dark:text-emerald-400'}>{hoveredCandle.open?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
                                                    <span>H: <span className="text-slate-700 dark:text-slate-300">{hoveredCandle.high?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
                                                    <span>L: <span className="text-slate-700 dark:text-slate-300">{hoveredCandle.low?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
                                                    <span>C: <span className={hoveredCandle.close >= (hoveredCandle.open || 0) ? 'text-emerald-500 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}>{hoveredCandle.close?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
                                                    {hoveredCandle.volume && <span>V: <span className="text-slate-700 dark:text-slate-300">{hoveredCandle.volume >= 1000 ? (hoveredCandle.volume / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 }) + 'k' : hoveredCandle.volume.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span></span>}
                                                </div>
                                            )}
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
                                                            <CoinIcon symbol={coin.symbol} size="sm" />
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
                                                                {percentChange(coin.price, coin.change24h).toFixed(2)}%
                                                            </span>
                                                        </div>
                                                    </button>
                                                ))
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Right Side: Controls */}
                            <div className="flex gap-2 items-center self-center flex-wrap justify-end">
                                {/* Chart Type Toggles */}
                                <div className="hidden md:flex bg-slate-100 dark:bg-slate-800/50 p-0.5 rounded-lg border border-slate-200 dark:border-slate-800/50">
                                    <button
                                        onClick={() => setChartType('area')}
                                        className={`p-1 rounded-md transition-all ${chartType === 'area' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                        title="Area Chart"
                                    >
                                        <Activity size={14} />
                                    </button>
                                    <button
                                        onClick={() => setChartType('candle')}
                                        className={`p-1 rounded-md transition-all ${chartType === 'candle' ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                                        title="Candlestick Chart"
                                    >
                                        <BarChart2 size={14} />
                                    </button>
                                </div>

                                <div className="h-3 w-px bg-slate-200 dark:bg-slate-700 mx-1"></div>

                                {/* Indicators Menu */}
                                <div className="relative" ref={indicatorMenuRef}>
                                    <button
                                        onClick={() => setShowIndicatorMenu(!showIndicatorMenu)}
                                        className={`flex items-center gap-1.5 px-2 py-1 text-[11px] font-bold rounded-lg border transition-all ${showIndicatorMenu
                                            ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800'
                                            : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800'
                                            }`}
                                    >
                                        <SlidersHorizontal size={12} />
                                        <span className="hidden sm:inline">Indicators</span>
                                        <ChevronDown size={10} className={`transition-transform ${showIndicatorMenu ? 'rotate-180' : ''}`} />
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

                                <div className="h-3 w-px bg-slate-200 dark:bg-slate-700 mx-1 hidden sm:block"></div>

                                <button
                                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                                    className="flex items-center justify-center w-6 h-6 rounded-full bg-white dark:bg-slate-800 shadow-sm border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-50 dark:hover:bg-slate-700 transition-all transform hover:scale-105 active:scale-95"
                                    title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
                                >
                                    {isSidebarOpen ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chart Content */}
                <div className="flex-1 relative flex flex-col bg-white dark:bg-slate-900 min-h-0">
                    <div className="flex-1 relative min-h-0">
                        {combinedChartData.length > 0 ? (
                            <LightweightChart
                                data={combinedChartData}
                                color={headerStats.change >= 0 ? '#10b981' : '#f43f5e'}
                                type={chartType}
                                indicators={indicators}
                                onChartClick={handleChartClick}
                                visibleRange={chartVisibleRange}
                                onLoadMore={handleLoadMore}
                                onCrosshairMove={setHoveredCandle}
                                resetKey={chartResetKey}
                            />
                        ) : (
                            <div className="flex h-full min-h-[400px] items-center justify-center bg-white dark:bg-slate-900">
                                <div className="flex items-center gap-2 text-sm font-medium text-slate-500 dark:text-slate-400">
                                    {loadingChart ? (
                                        <>
                                            <Loader2 className="animate-spin text-indigo-600" size={18} />
                                            <span>Loading chart...</span>
                                        </>
                                    ) : (
                                        <span>No chart data available.</span>
                                    )}
                                </div>
                            </div>
                        )}
                        {isRangeSelecting && (
                            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-indigo-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg z-10 pointer-events-none animate-in fade-in slide-in-from-top-2">
                                {rangeStart ? "Click end point..." : "Click start point..."}
                            </div>
                        )}
                    </div>
                    
                    {/* Timeframe selector (minimalist footer) */}
                    <div className="flex items-center justify-start gap-1 p-2 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50 flex-wrap shrink-0">
                        {/* Date Selection Tools */}
                        <div className="flex items-center gap-1">
                            <div className="relative" ref={datePickerRef}>
                                <button
                                    onClick={() => setShowDatePicker(!showDatePicker)}
                                    className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all flex items-center justify-center gap-1.5 ${showDatePicker || hasActiveRange
                                        ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400 shadow-sm border border-indigo-200 dark:border-indigo-800'
                                        : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800 border border-transparent'
                                        }`}
                                    title="Go to Date / Select Range"
                                >
                                    <Calendar size={14} />
                                    {hasActiveRange && <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse"></span>}
                                </button>

                                {/* Combined Popup */}
                                {showDatePicker && (
                                    <div className="absolute bottom-full left-0 mb-2 w-72 bg-white dark:bg-slate-900 rounded-xl shadow-xl border border-slate-200 dark:border-slate-800 z-[60] overflow-hidden animate-in fade-in zoom-in-95 duration-200 p-3">
                                        <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1 mb-3">
                                            <button 
                                                onClick={() => setDatePickerMode('single')}
                                                className={`flex-1 text-xs font-bold py-1.5 rounded-md transition-all ${datePickerMode === 'single' ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-200 shadow-sm' : 'text-slate-500 dark:text-slate-400'}`}
                                            >
                                                Go to Date
                                            </button>
                                            <button 
                                                onClick={() => setDatePickerMode('range')}
                                                className={`flex-1 text-xs font-bold py-1.5 rounded-md transition-all ${datePickerMode === 'range' ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-200 shadow-sm' : 'text-slate-500 dark:text-slate-400'}`}
                                            >
                                                Time Range
                                            </button>
                                        </div>

                                        {datePickerMode === 'single' ? (
                                            <div className="space-y-3">
                                                <div>
                                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Select Date & Time</label>
                                                    <input
                                                        type="datetime-local"
                                                        value={tempSingleDate}
                                                        onChange={(e) => {
                                                            setTempSingleDate(e.target.value);
                                                            if (e.target.value) {
                                                                const selectedTime = new Date(e.target.value).getTime();
                                                                const twelveHoursMs = 12 * 60 * 60 * 1000;
                                                                const start = Math.floor((selectedTime - twelveHoursMs) / 1000);
                                                                const end = Math.floor((selectedTime + twelveHoursMs) / 1000);
                                                                setChartVisibleRange({ from: start, to: end });
                                                                setHasActiveRange(true);
                                                                setTimeRange('Custom');
                                                            }
                                                        }}
                                                        className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                    />
                                                </div>
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
                                                            className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-2 text-[10px] font-mono text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                        />
                                                    </div>
                                                    <div>
                                                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">To</label>
                                                        <input
                                                            type="datetime-local"
                                                            value={manualDateTo}
                                                            onChange={(e) => setManualDateTo(e.target.value)}
                                                            className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-2 text-[10px] font-mono text-slate-700 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                                                        />
                                                    </div>
                                                </div>
                                                <button
                                                    onClick={() => { handleManualRangeApply(); setShowDatePicker(false); }}
                                                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2 rounded-lg transition-colors"
                                                >
                                                    Apply Range
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={() => {
                                    setIsRangeSelecting(true);
                                    setRangeStart(null);
                                    setShowDatePicker(false);
                                }}
                                className={`px-2.5 py-1 rounded-md transition-all flex items-center justify-center ${isRangeSelecting
                                    ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400 shadow-sm border border-indigo-200 dark:border-indigo-800 animate-pulse'
                                    : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800 border border-transparent'
                                    }`}
                                title="Select on Chart"
                            >
                                <Hand size={14} />
                            </button>
                            
                            {hasActiveRange && (
                                <button
                                    onClick={() => {
                                        clearRangeSelection();
                                        setShowDatePicker(false);
                                    }}
                                    className="px-2 py-1 text-rose-500 hover:text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-900/20 rounded-md transition-all flex items-center justify-center border border-transparent"
                                    title="Clear Selection"
                                >
                                    <X size={14} />
                                </button>
                            )}
                        </div>

                        <div className="h-4 w-px bg-slate-200 dark:bg-slate-700 mx-1"></div>

                        {['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'].map((range) => (
                            <button
                                key={range}
                                onClick={() => setTimeRange(range)}
                                className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${timeRange === range
                                    ? 'bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-sm border border-slate-200 dark:border-slate-600'
                                    : 'text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800 border border-transparent'
                                    }`}
                            >
                                {range}
                            </button>
                        ))}
                    </div>
                </div>
            </div>



            {/* RIGHT: Sidebar Panel (Market Intelligence) */}
            {
                isSidebarOpen && (
                    <div
                        ref={sidebarRef}
                        className="w-full md:w-80 lg:w-96 flex flex-col bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 h-[calc(100vh-3.5rem)] overflow-hidden transition-all duration-300"
                    >
                        

                        {/* Sidebar Content: Split View */}
                        <div className="flex-1 flex flex-col overflow-hidden min-h-0">

                            {/* TOP: News Context (50%) */}
                            <div className="flex-1 flex flex-col border-b border-slate-200 dark:border-slate-800 min-h-0 overflow-hidden">
                                <div className="p-3 bg-slate-50 dark:bg-slate-950/50 border-b border-slate-100 dark:border-slate-800 flex-none">
                                    <h4 className="font-bold text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                        <Globe size={14} className="text-blue-500" />
                                        Contextual News
                                    </h4>
                                </div>

                                <div ref={contextNewsListScrollRef} className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar min-h-0">
                                    {loadingNews ? (
                                        <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-2 py-8">
                                            <Loader2 size={20} className="animate-spin" />
                                            <p className="text-xs">Finding insights...</p>
                                        </div>
                                    ) : contextNews.length > 0 ? (
                                        contextNewsPageArticles.map((article) => (
                                            <div
                                                key={article.id}
                                                className={`p-3 bg-white dark:bg-slate-800 rounded-lg border transition-all group shadow-sm ring-1 ${getNewsCardAccent(article.sentiment)} ${getNewsCardHover(article.sentiment)} hover:ring-2`}
                                            >
                                                <div className="flex justify-between items-start mb-1.5">
                                                    <span className="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20 px-2 py-0.5 rounded-full">
                                                        {article.source}
                                                    </span>
                                                    <span className="text-[10px] text-slate-400">
                                                        {formatDateToLocalWithOffset(article.timestamp)}
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
                                            <p className="text-xs text-center">
                                                {hasActiveRange && chartVisibleRange
                                                    ? `No news related to ${selectedCoin.symbol} in the selected time range.`
                                                    : `No recent news related to ${selectedCoin.symbol}.`}
                                            </p>
                                        </div>
                                    )}
                                </div>

                                {!loadingNews && contextNewsTotalPages > 1 && (
                                    <div className="flex-none flex items-center justify-between gap-2 px-3 py-2 border-t border-slate-100 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/50">
                                        <button
                                            type="button"
                                            disabled={contextNewsSafePage <= 1}
                                            onClick={() => setContextNewsPage((p) => Math.max(1, p - 1))}
                                            className="flex items-center justify-center p-1.5 rounded-md text-slate-600 dark:text-slate-400 disabled:opacity-30 disabled:pointer-events-none hover:bg-white dark:hover:bg-slate-800 border border-transparent hover:border-slate-200 dark:hover:border-slate-700"
                                            aria-label="Previous page"
                                        >
                                            <ChevronLeft size={18} />
                                        </button>
                                        <span className="text-[10px] font-medium text-slate-500 dark:text-slate-400 tabular-nums">
                                            {contextNewsSafePage} / {contextNewsTotalPages}
                                            <span className="text-slate-400 dark:text-slate-500 font-normal"> · {contextNews.length} articles</span>
                                        </span>
                                        <button
                                            type="button"
                                            disabled={contextNewsSafePage >= contextNewsTotalPages}
                                            onClick={() => setContextNewsPage((p) => Math.min(contextNewsTotalPages, p + 1))}
                                            className="flex items-center justify-center p-1.5 rounded-md text-slate-600 dark:text-slate-400 disabled:opacity-30 disabled:pointer-events-none hover:bg-white dark:hover:bg-slate-800 border border-transparent hover:border-slate-200 dark:hover:border-slate-700"
                                            aria-label="Next page"
                                        >
                                            <ChevronRight size={18} />
                                        </button>
                                    </div>
                                )}
                            </div>

                            {/* BOTTOM: AI Forecast & Summary */}
                            <div className={`flex flex-col bg-slate-50 dark:bg-slate-950 overflow-hidden transition-all duration-300 ${showForecast ? 'flex-1 min-h-0' : 'flex-none'}`}>
                                <button
                                    onClick={() => setShowForecast(!showForecast)}
                                    className="p-3 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-white dark:bg-slate-900 flex-none hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors cursor-pointer w-full text-left"
                                >
                                    <h4 className="font-bold text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                                        <BrainCircuit size={14} className="text-purple-500" />
                                        AI Forecast
                                    </h4>
                                    <ChevronDown size={24} className={`text-slate-400 transition-transform duration-300 ${showForecast ? 'rotate-180' : ''}`} />
                                </button>
                                {showForecast && (
                                    <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                                        {!isAuthenticated ? (
                                            <div className="flex flex-col items-center justify-center h-full py-10 px-4 text-center text-slate-400">
                                                <ShieldAlert size={32} className="mb-3 opacity-20" />
                                                <h4 className="text-sm font-bold text-slate-500 dark:text-slate-400 mb-1">AI Summary Locked</h4>
                                                <p className="text-xs mb-4">Log in to unlock AI forecasts and summaries.</p>
                                                <NavLink
                                                    to="/login"
                                                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-bold transition-all shadow-md shadow-indigo-500/20"
                                                >
                                                    Login to Continue
                                                </NavLink>
                                            </div>
                                        ) : forecastResult ? (
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
                                )}
                            </div>

                        </div>
                    </div >
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
