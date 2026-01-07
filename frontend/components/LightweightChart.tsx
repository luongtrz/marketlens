import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { HistoryPoint } from '../types';

interface LightweightChartProps {
    data: HistoryPoint[];
    color?: string;
}

const LightweightChart: React.FC<LightweightChartProps> = ({ data, color = '#10b981' }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<any>(null);
    const candlestickSeriesRef = useRef<any>(null);
    const volumeSeriesRef = useRef<any>(null);

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

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
            },
        });

        // Candlestick Series
        const candlestickSeries = (chart as any).addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#f43f5e',
            borderUpColor: '#10b981',
            borderDownColor: '#f43f5e',
            wickUpColor: '#10b981',
            wickDownColor: '#f43f5e',
        });

        // Volume Series
        const volumeSeries = (chart as any).addHistogramSeries({
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
        candlestickSeriesRef.current = candlestickSeries;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update data when it changes
    useEffect(() => {
        if (!candlestickSeriesRef.current || !volumeSeriesRef.current || data.length === 0) return;

        const candlestickData: any[] = [];
        const volumeData: any[] = [];

        data.forEach((point) => {
            // Only process points with valid price data (skip forecast-only points)
            if (point.price !== undefined && point.ts) {
                const time = Math.floor(point.ts / 1000); // Convert to seconds

                // Candlestick data
                if (point.open !== undefined && point.high !== undefined && point.low !== undefined) {
                    candlestickData.push({
                        time,
                        open: point.open,
                        high: point.high,
                        low: point.low,
                        close: point.price,
                    });
                }

                // Volume data
                if (point.volume !== undefined) {
                    volumeData.push({
                        time,
                        value: point.volume,
                        color: point.price >= (point.open || point.price) ? '#10b98140' : '#f43f5e40',
                    });
                }
            }
        });

        // Set data
        if (candlestickData.length > 0) {
            candlestickSeriesRef.current.setData(candlestickData);
        }
        if (volumeData.length > 0) {
            volumeSeriesRef.current.setData(volumeData);
        }

        // Fit content
        chartRef.current?.timeScale().fitContent();
    }, [data]);

    return (
        <div
            ref={chartContainerRef}
            className="w-full h-full"
            style={{ minHeight: '400px' }}
        />
    );
};

export default LightweightChart;
