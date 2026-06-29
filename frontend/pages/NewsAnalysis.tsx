import React, { useEffect, useState, useMemo, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { NewsArticle } from '../types';
import { fetchLatestNews, fetchLatestNewsPaged, fetchNewsSourceHosts } from '../services/apiService';
import NewsCard from '../components/NewsCard';
import ArticleDetailModal from '../components/ArticleDetailModal';
import { Loader2, Filter, ArrowUpDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { formatSource } from '../utils/formatters';

const ITEMS_PER_PAGE = 20;

/** Coin tags from Supabase; keyword fallback is only for old/mock API payloads. */
function effectiveArticleCoins(a: NewsArticle): string[] {
  if (Array.isArray(a.coin) && a.coin.length > 0) return a.coin;
  if (a.tag) {
    if (a.tag === 'BTC & ETH') return ['BTC', 'ETH'];
    return [a.tag];
  }
  const hay = `${a.title} ${a.snippet}`.toLowerCase();
  const hasBtc = /\b(btc|bitcoin)\b/.test(hay);
  const hasEth = /\b(eth|ethereum|ether)\b/.test(hay)
    || /\b(vitalik|buterin|erc-?20|arbitrum|optimism|uniswap|aave|lido)\b/.test(hay);
  const tags = [];
  if (hasBtc) tags.push('BTC');
  if (hasEth) tags.push('ETH');
  return tags.length > 0 ? tags : ['General'];
}

function articleMatchesTag(article: NewsArticle, tag: string): boolean {
  if (tag === 'All') return true;
  return effectiveArticleCoins(article).includes(tag);
}

const NewsAnalysis: React.FC = () => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  /** When using server paging, equals Supabase-visible total rows (see API). Legacy: capped batch size after fetch. */
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [sourceHostOptions, setSourceHostOptions] = useState<string[]>([]);

  const [sentimentFilter, setSentimentFilter] = useState<string>('All');
  const [sourceFilter, setSourceFilter] = useState<string>('All');
  const [tagFilter, setTagFilter] = useState<string>('All');
  const [sortBy, setSortBy] = useState<string>('Latest');

  const [currentPage, setCurrentPage] = useState(1);

  const [selectedArticle, setSelectedArticle] = useState<NewsArticle | null>(null);

  const pageTopRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const navigate = useNavigate();
  const autoOpenRef = useRef<string | null>(null);

  const articleIdParam = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('articleId');
  }, [location.search]);

  /** True when filters allow PostgREST-backed paging (no client-only dimensions). */
  const canUseServerPaging = useMemo(
    () =>
      sentimentFilter === 'All' &&
      sortBy === 'Latest' &&
      (tagFilter === 'All' || tagFilter === 'BTC' || tagFilter === 'ETH' || tagFilter === 'General'),
    [sentimentFilter, sortBy, tagFilter],
  );

  useEffect(() => {
    let cancelled = false;
    const loadHosts = async () => {
      const hosts = await fetchNewsSourceHosts();
      if (!cancelled) setSourceHostOptions(hosts);
    };
    void loadHosts();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setCurrentPage(1);
  }, [sentimentFilter, sourceFilter, tagFilter, sortBy, canUseServerPaging]);

  useEffect(() => {
    if (canUseServerPaging) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const tagParam = tagFilter === 'All' ? undefined : tagFilter;
        const src = sourceFilter === 'All' ? undefined : sourceFilter;
        const data = await fetchLatestNews(undefined, undefined, tagParam, src);
        if (!cancelled) {
          setArticles(data);
          setTotalCount(data.length);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [canUseServerPaging, sentimentFilter, sourceFilter, tagFilter, sortBy]);

  useEffect(() => {
    if (!canUseServerPaging) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const tagParam = tagFilter === 'All' ? undefined : tagFilter;
        const src = sourceFilter === 'All' ? undefined : sourceFilter;
        const pg = await fetchLatestNewsPaged(currentPage, ITEMS_PER_PAGE, undefined, undefined, tagParam, src);
        if (!cancelled && pg) {
          setArticles(pg.items);
          setTotalCount(pg.total);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [canUseServerPaging, currentPage, tagFilter, sourceFilter]);

  useEffect(() => {
    if (!articleIdParam || selectedArticle || articles.length === 0) return;
    if (autoOpenRef.current === articleIdParam) return;
    const match = articles.find((article) => String(article.id) === String(articleIdParam));
    if (match) {
      autoOpenRef.current = articleIdParam;
      setSelectedArticle(match);
      navigate('/news', { replace: true });
    }
  }, [articleIdParam, articles, selectedArticle, navigate]);

  const uniqueSources = useMemo(() => {
    const sources = new Set<string>([...sourceHostOptions, ...articles.map((a) => a.source)]);
    return Array.from(sources).sort((a, b) => a.localeCompare(b));
  }, [articles, sourceHostOptions]);

  const filteredArticles = useMemo(() => {
    if (canUseServerPaging) {
      return articles;
    }
    return articles
      .filter((article) => {
        if (!articleMatchesTag(article, tagFilter)) {
          return false;
        }
        if (sentimentFilter !== 'All' && article.sentiment !== sentimentFilter) {
          return false;
        }
        if (sourceFilter !== 'All' && article.source !== sourceFilter) {
          return false;
        }
        // Server already filters by source when using paged fetch; keep for mixed/mock edge cases.
        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'SentimentScore') {
          return (b.sentimentScore || 0) - (a.sentimentScore || 0);
        }
        if (sortBy === 'Latest') {
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        }
        return 0;
      });
  }, [articles, canUseServerPaging, sentimentFilter, sourceFilter, tagFilter, sortBy]);

  const totalPages = useMemo(() => {
    if (canUseServerPaging) {
      return Math.max(1, Math.ceil(totalCount / ITEMS_PER_PAGE));
    }
    return Math.max(1, Math.ceil(filteredArticles.length / ITEMS_PER_PAGE));
  }, [canUseServerPaging, totalCount, filteredArticles.length]);

  const currentArticles = useMemo(() => {
    if (canUseServerPaging) {
      return filteredArticles;
    }
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredArticles.slice(startIndex, startIndex + ITEMS_PER_PAGE);
  }, [canUseServerPaging, filteredArticles, currentPage]);

  const rangeStart = useMemo(() => {
    if (currentArticles.length === 0) return 0;
    return (currentPage - 1) * ITEMS_PER_PAGE + 1;
  }, [currentArticles.length, currentPage]);

  const rangeEnd = useMemo(() => {
    if (currentArticles.length === 0) return 0;
    if (canUseServerPaging) {
      return Math.min(currentPage * ITEMS_PER_PAGE, totalCount);
    }
    return Math.min(currentPage * ITEMS_PER_PAGE, filteredArticles.length);
  }, [canUseServerPaging, currentArticles.length, currentPage, totalCount, filteredArticles.length]);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      pageTopRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleCardClick = (article: NewsArticle) => {
    setSelectedArticle(article);
  };

  const handleModalClose = () => {
    setSelectedArticle(null);
    if (articleIdParam) {
      navigate('/news', { replace: true });
    }
  };

  const countLabel = canUseServerPaging ? totalCount : filteredArticles.length;

  return (
    <div className="relative space-y-6 p-6 pb-14">
      <div ref={pageTopRef} className="pointer-events-none h-0 scroll-mt-[4.25rem]" aria-hidden />
      <header className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:justify-between md:items-end gap-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">News Intelligence</h2>
            <p className="text-slate-500 dark:text-slate-400">AI-aggregated news with sentiment scoring.</p>
          </div>

          <div className="flex flex-wrap gap-3 bg-white dark:bg-slate-900 p-3 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-400 dark:text-slate-300 uppercase tracking-wider">Tag:</span>
              <select
                value={tagFilter}
                onChange={(e) => setTagFilter(e.target.value)}
                className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
              >
                <option value="All">All Coins</option>
                <option value="BTC">BTC</option>
                <option value="ETH">ETH</option>
                <option value="General">General</option>
              </select>
            </div>

            <div className="w-px h-8 bg-slate-200 mx-1 hidden md:block"></div>

            <div className="flex items-center gap-2">
              <Filter size={14} className="text-slate-400 dark:text-slate-300" />
              <select
                value={sentimentFilter}
                onChange={(e) => setSentimentFilter(e.target.value)}
                className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
              >
                <option value="All">All Sentiments</option>
                <option value="Positive">Positive</option>
                <option value="Negative">Negative</option>
                <option value="Neutral">Neutral</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
              >
                <option value="All">All Sources</option>
                {uniqueSources.map((source) => (
                  <option key={source} value={source}>
                    {formatSource(source)}
                  </option>
                ))}
              </select>
            </div>

            <div className="w-px h-8 bg-slate-200 mx-1 hidden md:block dark:bg-slate-700"></div>

            <div className="flex items-center gap-2">
              <ArrowUpDown size={14} className="text-slate-400 dark:text-slate-300" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2 outline-none"
              >
                <option value="Latest">Sort: Latest</option>
                <option value="SentimentScore">Sort: Sentiment Score</option>
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
            <div className="text-center py-20 bg-slate-50 dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 border-dashed">
              <p className="text-slate-500 dark:text-slate-300">No articles match your filters.</p>
              <button
                onClick={() => {
                  setSentimentFilter('All');
                  setSourceFilter('All');
                  setTagFilter('All');
                }}
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

              <div className="flex justify-center items-center gap-4 mt-8 pt-4 border-t border-slate-200 dark:border-slate-800">
                <span className="text-xs text-slate-500 dark:text-slate-400 mr-2 tabular-nums">
                  {rangeStart}
                  –
                  {rangeEnd} of {countLabel}
                </span>
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="p-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={20} />
                </button>

                <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
                  Page <span className="text-indigo-600 font-bold">{currentPage}</span> of {totalPages}
                </span>

                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="p-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight size={20} />
                </button>
              </div>
            </>
          )}
        </>
      )}

      {selectedArticle && (
        <ArticleDetailModal article={selectedArticle} onClose={handleModalClose} />
      )}
    </div>
  );
};

export default NewsAnalysis;
