import React from 'react';
import { SignalType } from '../types';

interface TechnicalAnalysisProps {
  priceChange: number;
}

const TechnicalGauge: React.FC<{ title: string; value: number; signal: SignalType }> = ({ title, value, signal }) => {
  // Value expected between -1 (Strong Sell) and 1 (Strong Buy)
  
  const getSignalColor = (s: SignalType) => {
    switch (s) {
      case 'Strong Buy': return 'text-green-600';
      case 'Buy': return 'text-green-500';
      case 'Neutral': return 'text-slate-500';
      case 'Sell': return 'text-red-500';
      case 'Strong Sell': return 'text-red-600';
      default: return 'text-slate-500';
    }
  };

  const getPosition = (val: number) => {
    // Map -1 to 1 range to 0% to 100%
    return ((val + 1) / 2) * 100;
  };

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h4 className="text-sm font-medium text-slate-700">{title}</h4>
        <span className={`text-sm font-bold ${getSignalColor(signal)} uppercase`}>{signal}</span>
      </div>
      
      <div className="relative h-2 bg-slate-100 rounded-full mb-2">
        {/* Sections */}
        <div className="absolute top-0 left-0 w-1/5 h-full bg-red-600 rounded-l-full opacity-30"></div>
        <div className="absolute top-0 left-[20%] w-1/5 h-full bg-red-400 opacity-30"></div>
        <div className="absolute top-0 left-[40%] w-1/5 h-full bg-slate-400 opacity-30"></div>
        <div className="absolute top-0 left-[60%] w-1/5 h-full bg-green-400 opacity-30"></div>
        <div className="absolute top-0 left-[80%] w-1/5 h-full bg-green-600 rounded-r-full opacity-30"></div>

        {/* Marker */}
        <div 
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-slate-300 rounded-full shadow-md transition-all duration-500"
            style={{ left: `calc(${getPosition(value)}% - 8px)` }}
        ></div>
      </div>
      
      <div className="flex justify-between text-[10px] text-slate-400 uppercase font-medium mt-2">
        <span>Strong Sell</span>
        <span>Sell</span>
        <span>Neutral</span>
        <span>Buy</span>
        <span>Strong Buy</span>
      </div>
    </div>
  );
};

const TechnicalAnalysis: React.FC<TechnicalAnalysisProps> = ({ priceChange }) => {
  // Simulate indicators based on price change direction + some randomness
  const isBullish = priceChange > 0;
  
  // Randomize slightly to make it look realistic
  const summaryVal = isBullish ? 0.6 : -0.4;
  const oscillatorsVal = isBullish ? 0.2 : -0.1;
  const maVal = isBullish ? 0.8 : -0.7;

  const getSignal = (val: number): SignalType => {
      if (val > 0.6) return 'Strong Buy';
      if (val > 0.2) return 'Buy';
      if (val > -0.2) return 'Neutral';
      if (val > -0.6) return 'Sell';
      return 'Strong Sell';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <TechnicalGauge title="Summary" value={summaryVal} signal={getSignal(summaryVal)} />
      <TechnicalGauge title="Oscillators" value={oscillatorsVal} signal={getSignal(oscillatorsVal)} />
      <TechnicalGauge title="Moving Averages" value={maVal} signal={getSignal(maVal)} />
    </div>
  );
};

export default TechnicalAnalysis;