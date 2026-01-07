import React, { useEffect, useRef, memo } from 'react';

interface TradingViewWidgetProps {
  symbol: string;
}

const TradingViewWidget: React.FC<TradingViewWidgetProps> = ({ symbol }) => {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;

    // Clear previous widget content to prevent duplicates on re-render
    container.current.innerHTML = '';

    // Create a container specifically for the widget script to populate
    const widgetDiv = document.createElement('div');
    widgetDiv.className = 'tradingview-widget-container__widget';
    widgetDiv.style.height = '100%';
    widgetDiv.style.width = '100%';
    container.current.appendChild(widgetDiv);

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js";
    script.type = "text/javascript";
    script.async = true;

    // Map the simple symbol (e.g., BTC) to TradingView format (e.g., COINBASE:BTCUSD)
    const tvSymbol = `COINBASE:${symbol}USD`;

    const widgetConfig = {
      symbols: [
        [
          symbol,
          tvSymbol + "|1D"
        ]
      ],
      chartOnly: false,
      width: "100%",
      height: "100%",
      locale: "en",
      colorTheme: "light",
      autosize: true,
      showVolume: false,
      hideDateRanges: false,
      scalePosition: "right",
      scaleMode: "Normal",
      fontFamily: "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
      noTimeScale: false,
      valuesTracking: "1",
      changeMode: "price-and-percent",
      chartType: "area",
      maLineColor: "#2962FF",
      maLineWidth: 1,
      maLength: 9,
      lineWidth: 2,
      lineType: 0,
      dateRanges: [
        "1d|1",
        "1m|30",
        "3m|60",
        "12m|1D",
        "60m|1W",
        "all|1M"
      ],
      // Theme Customization to match app (Light Mode)
      backgroundColor: "rgba(255, 255, 255, 0)", // Transparent
      gridLineColor: "rgba(0, 0, 0, 0.05)",
      fontColor: "#64748b", // Slate 500
      widgetFontColor: "#64748b",
      upColor: "#10b981", // Emerald 500
      downColor: "#ef4444", // Red 500
      borderUpColor: "#10b981",
      borderDownColor: "#ef4444",
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    };

    script.innerHTML = JSON.stringify(widgetConfig);
    container.current.appendChild(script);
  }, [symbol]);

  return (
    <div className="tradingview-widget-container h-full w-full overflow-hidden rounded-xl" ref={container} />
  );
};

export default memo(TradingViewWidget);