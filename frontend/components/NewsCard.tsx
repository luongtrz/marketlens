import React from 'react';
import { NewsArticle } from '../types';
import { BrainCircuit, TrendingUp, TrendingDown, Minus, ExternalLink } from 'lucide-react';

import { formatSource, formatDateToLocalWithOffset, formatUnitSentiment } from '../utils/formatters';

interface NewsCardProps {
  article: NewsArticle;
  onClick: (article: NewsArticle) => void;
}

const NewsCard: React.FC<NewsCardProps> = ({ article, onClick }) => {

  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return <TrendingUp className="w-5 h-5 text-sky-600" />;
      case 'Negative': return <TrendingDown className="w-5 h-5 text-violet-600" />;
      default: return <Minus className="w-5 h-5 text-slate-500" />;
    }
  };

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return 'bg-emerald-600 border-emerald-600 text-white';
      case 'Negative': return 'bg-red-600 border-red-600 text-white';
      default: return 'bg-amber-500 border-amber-500 text-white';
    }
  };

  const getSentimentAccent = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return 'border-emerald-300 ring-emerald-100';
      case 'Negative': return 'border-red-300 ring-red-100';
      default: return 'border-amber-300 ring-amber-100';
    }
  };

  const getSentimentHover = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return 'hover:ring-emerald-300 hover:shadow-emerald-200/40';
      case 'Negative': return 'hover:ring-red-300 hover:shadow-red-200/40';
      default: return 'hover:ring-amber-300 hover:shadow-amber-200/40';
    }
  };

  const getScoreColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return 'text-emerald-600';
      case 'Negative': return 'text-red-600';
      default: return 'text-amber-600';
    }
  };

  return (
    <div
      onClick={() => onClick(article)}
      className={`bg-white dark:bg-slate-900 border rounded-xl p-5 hover:shadow-lg transition-all cursor-pointer group ring-1 ${getSentimentAccent(article.sentiment)} ${getSentimentHover(article.sentiment)} hover:ring-2`}
    >
      <div className="flex justify-between items-start mb-3">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-300 px-2 py-1 bg-slate-50 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700">{formatSource(article.source)} • {formatDateToLocalWithOffset(article.timestamp)}</span>
        {article.sentiment && (
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border shadow-sm ${getSentimentColor(article.sentiment)}`}>
            {getSentimentIcon(article.sentiment)}
            <span className="text-xs font-extrabold uppercase tracking-wide">{article.sentiment}</span>
          </div>
        )}
      </div>

      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2 leading-snug group-hover:text-indigo-600 transition-colors">
        {article.title}
      </h3>

      <p className="text-slate-600 dark:text-slate-300 text-sm mb-4 line-clamp-2">
        {article.snippet}
      </p>

      {/* Mini Footer */}
      <div className="flex items-center justify-between mt-auto pt-3 border-t border-slate-100 dark:border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          {article.sentimentScore !== undefined && (
            <span className={getScoreColor(article.sentiment)}>
              Sentiment: {formatUnitSentiment(article.sentimentScore)}{' '}
              <span className="text-slate-400 dark:text-slate-500 font-normal">(-1…+1)</span>
            </span>
          )}
        </div>
        <button className="flex items-center gap-1.5 text-xs font-medium text-indigo-600 group-hover:text-indigo-700 transition-colors">
          <BrainCircuit size={14} />
          AI Analysis
        </button>
      </div>
    </div>
  );
};

export default NewsCard;