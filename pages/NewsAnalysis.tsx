import React, { useEffect, useState, useMemo } from 'react';
import { NewsArticle } from '../types';
import { fetchLatestNews } from '../services/geminiService';
import NewsCard from '../components/NewsCard';
import ArticleDetailModal from '../components/ArticleDetailModal';
import { Loader2, Filter, ArrowUpDown, ChevronLeft, ChevronRight } from 'lucide-react';

const ITEMS_PER_PAGE = 6;

const NewsAnalysis: React.FC = () => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);

  // Filter States
  const [sentimentFilter, setSentimentFilter] = useState<string>('All');
  const [impactFilter, setImpactFilter] = useState<string>('All');
  const [sourceFilter, setSourceFilter] = useState<string>('All');
  const [sortBy, setSortBy] = useState<string>('Latest');

  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);

  // Modal State
  const [selectedArticle, setSelectedArticle] = useState<NewsArticle | null>(null);

  useEffect(() => {
    const loadNews = async () => {
      try {
        const data = await fetchLatestNews();
        setArticles(data);
      } finally {
        setLoading(false);
      }
    };
    loadNews();
  }, []);

  // Compute Unique Sources
  const uniqueSources = useMemo(() => {
    const sources = new Set(articles.map(a => a.source));
    return Array.from(sources);
  }, [articles]);

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [sentimentFilter, impactFilter, sourceFilter, sortBy]);

  // Filter and Sort Logic
  const filteredArticles = useMemo(() => {
    return articles
      .filter(article => {
        // Sentiment Filter
        if (sentimentFilter !== 'All' && article.sentiment !== sentimentFilter) {
          return false;
        }
        // Source Filter
        if (sourceFilter !== 'All' && article.source !== sourceFilter) {
            return false;
        }
        // Impact Filter
        if (impactFilter !== 'All') {
          const score = article.impactScore || 0;
          if (impactFilter === 'High' && score < 70) return false;
          if (impactFilter === 'Medium' && (score < 30 || score >= 70)) return false;
          if (impactFilter === 'Low' && score >= 30) return false;
        }
        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'Impact') {
          return (b.impactScore || 0) - (a.impactScore || 0);
        }
        // Default to keeping original order (Latest based on mock fetch)
        return 0; 
      });
  }, [articles, sentimentFilter, impactFilter, sourceFilter, sortBy]);

  // Pagination Logic
  const totalPages = Math.ceil(filteredArticles.length / ITEMS_PER_PAGE);
  const currentArticles = useMemo(() => {
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredArticles.slice(startIndex, startIndex + ITEMS_PER_PAGE);
  }, [filteredArticles, currentPage]);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleCardClick = (article: NewsArticle) => {
    setSelectedArticle(article);
  };

  return (
    <div className="space-y-6 relative p-6">
      <header className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-4">
            <div>
            <h2 className="text-2xl font-bold text-slate-900">News Intelligence</h2>
            <p className="text-slate-500">AI-aggregated news with sentiment scoring.</p>
            </div>
            
            {/* Filter Controls */}
            <div className="flex flex-wrap gap-3 bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            
            {/* Sentiment Filter */}
            <div className="flex items-center gap-2">
                <Filter size={14} className="text-slate-400" />
                <select 
                value={sentimentFilter}
                onChange={(e) => setSentimentFilter(e.target.value)}
                className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
                >
                <option value="All">All Sentiments</option>
                <option value="Positive">Positive</option>
                <option value="Negative">Negative</option>
                <option value="Neutral">Neutral</option>
                </select>
            </div>

            {/* Source Filter */}
            <div className="flex items-center gap-2">
                <select 
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
                >
                <option value="All">All Sources</option>
                {uniqueSources.map(source => (
                    <option key={source} value={source}>{source}</option>
                ))}
                </select>
            </div>

            {/* Impact Filter */}
            <div className="flex items-center gap-2">
                <select 
                value={impactFilter}
                onChange={(e) => setImpactFilter(e.target.value)}
                className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
                >
                <option value="All">All Impacts</option>
                <option value="High">High Impact (&gt;70)</option>
                <option value="Medium">Medium Impact</option>
                <option value="Low">Low Impact (&lt;30)</option>
                </select>
            </div>

            <div className="w-px h-8 bg-slate-200 mx-1 hidden md:block"></div>

            {/* Sort Control */}
            <div className="flex items-center gap-2">
                <ArrowUpDown size={14} className="text-slate-400" />
                <select 
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
                >
                    <option value="Latest">Sort: Latest</option>
                    <option value="Impact">Sort: Impact</option>
                </select>
            </div>

            </div>
        </div>
      </header>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="animate-spin text-indigo-600 w-10 h-10" />
        </div>
      ) : (
        <>
          {filteredArticles.length === 0 ? (
            <div className="text-center py-20 bg-slate-50 rounded-2xl border border-slate-200 border-dashed">
               <p className="text-slate-500">No articles match your filters.</p>
               <button 
                  onClick={() => { setSentimentFilter('All'); setImpactFilter('All'); setSourceFilter('All'); }}
                  className="mt-2 text-indigo-600 hover:text-indigo-500 text-sm font-medium"
               >
                  Clear Filters
               </button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
                {currentArticles.map((article) => (
                  <NewsCard key={article.id} article={article} onClick={handleCardClick} />
                ))}
              </div>

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div className="flex justify-center items-center gap-4 mt-8 pt-4 border-t border-slate-200">
                  <button 
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="p-2 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft size={20} />
                  </button>
                  
                  <span className="text-sm font-medium text-slate-600">
                    Page <span className="text-indigo-600 font-bold">{currentPage}</span> of {totalPages}
                  </span>

                  <button 
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="p-2 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronRight size={20} />
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* Detailed Analysis Modal */}
      {selectedArticle && (
          <ArticleDetailModal 
            article={selectedArticle} 
            onClose={() => setSelectedArticle(null)} 
          />
      )}
    </div>
  );
};

export default NewsAnalysis;