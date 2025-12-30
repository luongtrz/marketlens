import React from 'react';
import { NewsArticle } from '../types';
import { BrainCircuit, TrendingUp, TrendingDown, Minus, ExternalLink } from 'lucide-react';

interface NewsCardProps {
  article: NewsArticle;
  onClick: (article: NewsArticle) => void;
}

const NewsCard: React.FC<NewsCardProps> = ({ article, onClick }) => {
  
  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return <TrendingUp className="w-5 h-5 text-green-600" />;
      case 'Negative': return <TrendingDown className="w-5 h-5 text-red-600" />;
      default: return <Minus className="w-5 h-5 text-slate-400" />;
    }
  };

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'Positive': return 'bg-green-100 border-green-200 text-green-700';
      case 'Negative': return 'bg-red-100 border-red-200 text-red-700';
      default: return 'bg-slate-100 border-slate-200 text-slate-600';
    }
  };

  return (
    <div 
      onClick={() => onClick(article)}
      className="bg-white border border-slate-200 rounded-xl p-5 hover:border-indigo-400 hover:shadow-lg hover:shadow-indigo-500/5 transition-all cursor-pointer group"
    >
      <div className="flex justify-between items-start mb-3">
        <span className="text-xs font-medium text-slate-500 px-2 py-1 bg-slate-50 rounded border border-slate-200">{article.source} • {article.timestamp}</span>
        {article.sentiment && (
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${getSentimentColor(article.sentiment)}`}>
            {getSentimentIcon(article.sentiment)}
            <span className="text-xs font-bold uppercase">{article.sentiment}</span>
          </div>
        )}
      </div>

      <h3 className="text-lg font-semibold text-slate-900 mb-2 leading-snug group-hover:text-indigo-600 transition-colors">
        {article.title}
      </h3>
      
      <p className="text-slate-600 text-sm mb-4 line-clamp-2">
        {article.snippet}
      </p>

      {/* Mini Footer */}
      <div className="flex items-center justify-between mt-auto pt-3 border-t border-slate-100">
         <div className="flex items-center gap-2 text-xs text-slate-500">
             {article.impactScore !== undefined && (
                 <span className={`${article.impactScore > 70 ? 'text-orange-600' : 'text-slate-500'}`}>
                    Impact Score: {article.impactScore}
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