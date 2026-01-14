import React, { useEffect, useRef, useState } from 'react';
import { createChart, CandlestickSeries, HistogramSeries, AreaSeries, LineSeries } from 'lightweight-charts';
import { HistoryPoint } from '../types';
import { calculateBollingerBands, getRSIValues, calculateMACD } from '../utils/indicators';

interface LightweightChartProps {
    data: HistoryPoint[];
    color?: string;
    type?: 'candle' | 'area';
    indicators?: {
        rsi: boolean;
        macd: boolean;
        bollinger: boolean;
    };
    visibleRange?: { from: number; to: number } | null;
    onChartClick?: (time: number) => void;
}

const LightweightChart: React.FC<LightweightChartProps> = ({
    data,
    color = '#10b981',
    type = 'candle',
    indicators = { rsi: false, macd: false, bollinger: false },
    visibleRange = null,
    onChartClick
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<any>(null);
    const mainSeriesRef = useRef<any>(null);
    const volumeSeriesRef = useRef<any>(null);
    const onChartClickRef = useRef(onChartClick);
    const rangeMarkersRef = useRef<any[]>([]);
    const isInitializedRef = useRef(false); // Track if chart has been initially fitted
    const prevTypeRef = useRef(type);
    const prevIndicatorsRef = useRef(indicators);

    // Update handler ref when prop changes to avoid stale closures
    useEffect(() => {
        onChartClickRef.current = onChartClick;
    }, [onChartClick]);

    // Indicator Refs
    const bbUpperRef = useRef<any>(null);
    const bbLowerRef = useRef<any>(null);
    const rsiSeriesRef = useRef<any>(null);
    const macdSeriesRef = useRef<any>(null);
    const macdSignalRef = useRef<any>(null);
    const macdHistRef = useRef<any>(null);

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Reset refs
        mainSeriesRef.current = null;
        volumeSeriesRef.current = null;
        bbUpperRef.current = null;
        bbLowerRef.current = null;
        rsiSeriesRef.current = null;
        macdSeriesRef.current = null;
        macdSignalRef.current = null;
        macdHistRef.current = null;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: 'transparent' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: '#e5e7eb20' },
                horzLines: { color: '#e5e7eb20' },
            },
            crosshair: {
                mode: 1 as any,
                vertLine: {
                    width: 1,
                    color: '#94a3b8',
                    style: 3 as any,
                },
                horzLine: {
                    width: 1,
                    color: '#94a3b8',
                    style: 3 as any,
                },
            },
            rightPriceScale: {
                borderColor: '#e5e7eb40',
            },
            timeScale: {
                borderColor: '#e5e7eb40',
                timeVisible: true,
                secondsVisible: false,
                fixLeftEdge: true,
                shiftVisibleRangeOnNewBar: true,
                rightOffset: 12,
            },
            // Enable zoom and pan
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: false,
            },
            handleScale: {
                mouseWheel: true,
                pinch: true,
                axisPressedMouseMove: true,
                axisDoubleClickReset: true,
            },
        });

        // Initialize Volume Series (always present)
        const volumeSeries = chart.addSeries(HistogramSeries, {
            color: '#94a3b8',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        });

        chartRef.current = chart;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize with ResizeObserver
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        const resizeObserver = new ResizeObserver(() => {
            handleResize();
        });

        if (chartContainerRef.current) {
            resizeObserver.observe(chartContainerRef.current);
        }

        // Click Listener
        chart.subscribeClick((param) => {
            if (!onChartClickRef.current) return;

            let time = param.time as number;
            // Fallback for empty space clicks
            if (!time && param.point) {
                time = chart.timeScale().coordinateToTime(param.point.x) as number;
            }
            if (time) {
                onChartClickRef.current(time);
            }
        });

        return () => {
            resizeObserver.disconnect();
            chart.remove();
            chartRef.current = null;
            mainSeriesRef.current = null;
            volumeSeriesRef.current = null;
            bbUpperRef.current = null;
            bbLowerRef.current = null;
            rsiSeriesRef.current = null;
            macdSeriesRef.current = null;
            macdSignalRef.current = null;
            macdHistRef.current = null;
        };
    }, []);

    // Handle Visible Range Changes
    const [overlayStyle, setOverlayStyle] = useState<React.CSSProperties>({ display: 'none' });

    // Handle Visible Range Changes (Highlight Overlay)
    useEffect(() => {
        if (!chartRef.current) return;

        const timeScale = chartRef.current.timeScale();

        const updateOverlay = () => {
            if (!visibleRange || !visibleRange.from || !visibleRange.to) {
                setOverlayStyle({ display: 'none' });
                return;
            }

            const fromX = timeScale.timeToCoordinate(visibleRange.from);
            const toX = timeScale.timeToCoordinate(visibleRange.to);

            // Handle cases where points might be off-screen (null)
            // If one is visible and one is not, we might still want to show it?
            // For simplicity, if either is null, we try to estimate or hide?
            // Lightweight Charts returns null if the time is not in the visible logical range?
            // Actually it returns null if time is not found or not visible.

            if (fromX === null || toX === null) {
                setOverlayStyle({ display: 'none' });
                return;
            }

            setOverlayStyle({
                position: 'absolute',
                left: Math.min(fromX, toX),
                width: Math.abs(toX - fromX),
                top: 0,
                bottom: 30, // Leave space for time scale
                backgroundColor: 'rgba(99, 102, 241, 0.2)', // Indigo-500 with 20% opacity
                borderLeft: '1px dashed #6366f1',
                borderRight: '1px dashed #6366f1',
                pointerEvents: 'none',
                display: 'block',
                zIndex: 2, // Above chart canvas (which is usually z=1 or 0)
            });
        };

        // Initial update
        updateOverlay();

        // Subscribe to changes (panning/zooming) to keep overlay in sync
        timeScale.subscribeVisibleTimeRangeChange(updateOverlay);

        return () => {
            timeScale.unsubscribeVisibleTimeRangeChange(updateOverlay);
        };
    }, [visibleRange]);

    // Handle Type Change & Data Update & Indicators
    useEffect(() => {
        if (!chartRef.current || !volumeSeriesRef.current) return;

        // 1. Manage Main Series (Candle/Area)
        if (mainSeriesRef.current) {
            chartRef.current.removeSeries(mainSeriesRef.current);
            mainSeriesRef.current = null;
        }

        if (type === 'candle') {
            mainSeriesRef.current = chartRef.current.addSeries(CandlestickSeries, {
                upColor: '#10b981',
                downColor: '#f43f5e',
                borderUpColor: '#10b981',
                borderDownColor: '#f43f5e',
                wickUpColor: '#10b981',
                wickDownColor: '#f43f5e',
            });
        } else {
            mainSeriesRef.current = chartRef.current.addSeries(AreaSeries, {
                lineColor: color,
                topColor: `${color}40`,
                bottomColor: `${color}00`,
                lineWidth: 2,
            });
        }

        // 2. Manage Indicators Cleanup
        if (bbUpperRef.current) { chartRef.current.removeSeries(bbUpperRef.current); bbUpperRef.current = null; }
        if (bbLowerRef.current) { chartRef.current.removeSeries(bbLowerRef.current); bbLowerRef.current = null; }
        if (rsiSeriesRef.current) { chartRef.current.removeSeries(rsiSeriesRef.current); rsiSeriesRef.current = null; }
        if (macdSeriesRef.current) { chartRef.current.removeSeries(macdSeriesRef.current); macdSeriesRef.current = null; }
        if (macdSignalRef.current) { chartRef.current.removeSeries(macdSignalRef.current); macdSignalRef.current = null; }
        if (macdHistRef.current) { chartRef.current.removeSeries(macdHistRef.current); macdHistRef.current = null; }

        // 3. Process Data
        if (data.length === 0) return;

        const mainData: any[] = [];
        const volumeData: any[] = [];

        data.forEach((point) => {
            if (point.price !== undefined && point.ts) {
                const time = Math.floor(point.ts / 1000);

                if (type === 'candle') {
                    if (point.open !== undefined && point.high !== undefined && point.low !== undefined) {
                        mainData.push({
                            time,
                            open: point.open,
                            high: point.high,
                            low: point.low,
                            close: point.price,
                        });
                    }
                } else {
                    mainData.push({ time, value: point.price });
                }

                if (point.volume !== undefined) {
                    volumeData.push({
                        time,
                        value: point.volume,
                        color: point.price >= (point.open || point.price) ? '#10b98140' : '#f43f5e40',
                    });
                }
            }
        });

        // Set Main & Volume Data
        if (mainData.length > 0 && mainSeriesRef.current) mainSeriesRef.current.setData(mainData);
        if (volumeData.length > 0) volumeSeriesRef.current.setData(volumeData);


        // 4. Render Indicators

        // Bollinger Bands (Overlay)
        if (indicators.bollinger) {
            const bands = calculateBollingerBands(data);
            bbUpperRef.current = chartRef.current.addSeries(LineSeries, { color: '#60a5fa', lineWidth: 1, title: 'BB Upper' });
            bbLowerRef.current = chartRef.current.addSeries(LineSeries, { color: '#60a5fa', lineWidth: 1, title: 'BB Lower' });

            bbUpperRef.current.setData(bands.map(b => ({ time: b.time, value: b.upper })));
            bbLowerRef.current.setData(bands.map(b => ({ time: b.time, value: b.lower })));
        }

        // RSI (Separate Pane - faked with margins)
        if (indicators.rsi) {
            // If RSI is enabled, squeeze main chart to top 70% and RSI to bottom 25%
            mainSeriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.3 } });

            const rsiData = getRSIValues(data);
            rsiSeriesRef.current = chartRef.current.addSeries(LineSeries, {
                color: '#8b5cf6',
                lineWidth: 2,
                priceScaleId: 'rsi',
                title: 'RSI (14)'
            });

            rsiSeriesRef.current.priceScale().applyOptions({
                scaleMargins: { top: 0.75, bottom: 0 },
                borderVisible: false,
            });

            rsiSeriesRef.current.setData(rsiData);
        } else if (indicators.macd) {
            // If MACD enabled (and no RSI - simplistic logic for now to avoid collision)
            mainSeriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.3 } });

            const macdData = calculateMACD(data);

            macdHistRef.current = chartRef.current.addSeries(HistogramSeries, { color: '#94a3b8', priceScaleId: 'macd' });
            macdSeriesRef.current = chartRef.current.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 1, priceScaleId: 'macd' });
            macdSignalRef.current = chartRef.current.addSeries(LineSeries, { color: '#f97316', lineWidth: 1, priceScaleId: 'macd' });

            chartRef.current.priceScale('macd').applyOptions({
                scaleMargins: { top: 0.75, bottom: 0 },
                borderVisible: false,
            });

            macdHistRef.current.setData(macdData.map(d => ({
                time: d.time,
                value: d.histogram,
                color: d.histogram >= 0 ? '#22c55e' : '#ef4444'
            })));
            macdSeriesRef.current.setData(macdData.map(d => ({ time: d.time, value: d.macd })));
            macdSignalRef.current.setData(macdData.map(d => ({ time: d.time, value: d.signal })));
        } else {
            // Reset main chart to full height (minus volume)
            mainSeriesRef.current.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.15 } });
        }

        // Fit content on: 1) Initial load, 2) Type change, 3) Indicator change
        // But NOT on real-time data updates (preserves user's zoom)
        const typeChanged = prevTypeRef.current !== type;
        const indicatorsChanged =
            prevIndicatorsRef.current.rsi !== indicators.rsi ||
            prevIndicatorsRef.current.macd !== indicators.macd ||
            prevIndicatorsRef.current.bollinger !== indicators.bollinger;

        if (!isInitializedRef.current || typeChanged || indicatorsChanged) {
            chartRef.current.timeScale().fitContent();
            isInitializedRef.current = true;
            prevTypeRef.current = type;
            prevIndicatorsRef.current = indicators;
        }

    }, [data, type, color, indicators]);

    // Highlight selected range with vertical markers
    useEffect(() => {
        if (!chartRef.current || !mainSeriesRef.current) return;

        // Clear previous markers
        rangeMarkersRef.current.forEach(line => {
            if (line && line.remove) line.remove();
        });
        rangeMarkersRef.current = [];

    }, [visibleRange]);

    return (
        <div className="relative w-full h-full" style={{ minHeight: '400px' }}>
            <div
                ref={chartContainerRef}
                className="w-full h-full"
            />
            <div style={overlayStyle} />
        </div>
    );
};

export default LightweightChart;
