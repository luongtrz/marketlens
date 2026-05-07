import React from 'react';
import { NewsArticle } from '../types';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

import { formatSource, formatDateToLocalWithOffset, formatUnitSentiment, polarityFromUnitScore } from '../utils/formatters';

function resolveCardSentiment(article: NewsArticle): 'Positive' | 'Negative' | 'Neutral' | null {
  const raw = article.sentiment;
  if (raw !== undefined && raw !== null && String(raw).trim() !== '') {
    const t = String(raw).toLowerCase();
    if (t === 'positive' || t === 'bullish') return 'Positive';
    if (t === 'negative' || t === 'bearish') return 'Negative';
    return 'Neutral';
  }
  if (article.sentimentScore !== undefined && Number.isFinite(article.sentimentScore)) {
    return polarityFromUnitScore(article.sentimentScore);
  }
  return null;
}

interface NewsCardProps {
  article: NewsArticle;
  onClick: (article: NewsArticle) => void;
}

const NewsCard: React.FC<NewsCardProps> = ({ article, onClick }) => {
  const label = resolveCardSentiment(article);

  const getSentimentIcon = (sentiment: 'Positive' | 'Negative' | 'Neutral') => {
    const shadow = 'drop-shadow-[0_1px_1px_rgba(0,0,0,0.45)]';
    switch (sentiment) {
      case 'Positive':
        return (
          <TrendingUp
            className={`w-5 h-5 shrink-0 text-amber-100 ${shadow}`}
            strokeWidth={2.5}
            aria-hidden
          />
        );
      case 'Negative':
        return (
          <TrendingDown
            className={`w-5 h-5 shrink-0 text-sky-100 ${shadow}`}
            strokeWidth={2.5}
            aria-hidden
          />
        );
      default:
        return <Minus className={`w-5 h-5 shrink-0 text-white ${shadow}`} strokeWidth={2.5} aria-hidden />;
    }
  };

  const getSentimentColor = (sentiment: 'Positive' | 'Negative' | 'Neutral') => {
    switch (sentiment) {
      // Nền đặc trùng màu viền (như Neutral), không dùng nền trắng / alpha làm nhạt
      case 'Positive':
        return 'bg-teal-700 border-teal-700 text-white shadow-sm dark:bg-teal-700 dark:border-teal-600';
      case 'Negative':
        return 'bg-rose-700 border-rose-700 text-white shadow-sm dark:bg-rose-700 dark:border-rose-600';
      default:
        return 'bg-amber-600 border-amber-600 text-white dark:bg-amber-600 dark:border-amber-500';
    }
  };

  /** Dòng Sentiment footer: cùng tông pill với badge góc trên */
  const getFooterSentimentPillClass = (sentiment: 'Positive' | 'Negative' | 'Neutral' | null) => {
    const s = sentiment ?? 'Neutral';
    switch (s) {
      case 'Positive':
        return 'bg-teal-700 border border-teal-600 text-white dark:bg-teal-700 dark:border-teal-600';
      case 'Negative':
        return 'bg-rose-700 border border-rose-600 text-white dark:bg-rose-700 dark:border-rose-600';
      default:
        return 'bg-amber-600 border border-amber-500 text-white dark:bg-amber-600 dark:border-amber-500';
    }
  };

  const getSentimentAccent = (sentiment: 'Positive' | 'Negative' | 'Neutral') => {
    switch (sentiment) {
      case 'Positive': return 'border-teal-200/70 ring-teal-100/40 dark:border-teal-700/55 dark:ring-teal-950/30';
      case 'Negative': return 'border-rose-200/70 ring-rose-100/40 dark:border-rose-700/55 dark:ring-rose-950/30';
      default: return 'border-amber-200/70 ring-amber-100/40 dark:border-amber-700/50 dark:ring-amber-950/30';
    }
  };

  const getSentimentHover = (sentiment: 'Positive' | 'Negative' | 'Neutral') => {
    switch (sentiment) {
      case 'Positive': return 'hover:ring-teal-300/70 hover:shadow-teal-900/15 dark:hover:ring-teal-600/40';
      case 'Negative': return 'hover:ring-rose-300/70 hover:shadow-rose-900/15 dark:hover:ring-rose-600/40';
      default: return 'hover:ring-amber-300 hover:shadow-amber-200/40 dark:hover:ring-amber-600/50';
    }
  };

  const accentSentiment = label ?? 'Neutral';

  return (
    <div
      onClick={() => onClick(article)}
      className={`bg-white dark:bg-slate-900 border rounded-xl p-5 hover:shadow-lg transition-all cursor-pointer group ring-1 ${getSentimentAccent(accentSentiment)} ${getSentimentHover(accentSentiment)} hover:ring-2`}
    >
      <div className="flex justify-between items-start mb-3">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-300 px-2 py-1 bg-slate-50 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700">{formatSource(article.source)} • {formatDateToLocalWithOffset(article.timestamp)}</span>
        {label !== null ? (
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border shadow-sm ${getSentimentColor(label)}`}>
            {getSentimentIcon(label)}
            <span className="text-xs font-extrabold uppercase tracking-wide text-white">{label}</span>
          </div>
        ) : null}
      </div>

      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2 leading-snug group-hover:text-indigo-600 transition-colors">
        {article.title}
      </h3>

      <p className="text-slate-600 dark:text-slate-300 text-sm mb-4 line-clamp-2">
        {article.snippet}
      </p>

      {article.sentimentScore !== undefined && (
        <div className="mt-auto pt-3 border-t border-slate-100 dark:border-slate-800">
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold tabular-nums ${getFooterSentimentPillClass(
              label ?? polarityFromUnitScore(article.sentimentScore),
            )}`}
          >
            Sentiment: {formatUnitSentiment(article.sentimentScore)}
          </span>
        </div>
      )}
    </div>
  );
};

export default NewsCard;