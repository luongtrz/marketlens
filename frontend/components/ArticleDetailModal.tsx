import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { NavLink, useLocation } from 'react-router-dom';
import { NewsArticle, AnalysisStatus } from '../types';
import { analyzeArticle, askNewsContext } from '../services/apiService';
import { Loader2, X, BrainCircuit, Sparkles, Send, ExternalLink } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { formatDateToLocalWithOffset, formatUnitSentiment, polarityFromUnitScore } from '../utils/formatters';
import NewsSourceLogo from './NewsSourceLogo';

interface ArticleDetailModalProps {
    article: NewsArticle;
    onClose: () => void;
}

/** Keep Supabase/list sentiment; merge only summary fields from ``/ai/analyze-article``. */
function mergeArticleAnalysis(prev: NewsArticle, listArticle: NewsArticle, analysis: Partial<NewsArticle>): NewsArticle {
    const listScore = listArticle.sentimentScore;
    const analyzedScore = analysis.sentimentScore;
    const sentimentScore =
        typeof listScore === 'number' && Number.isFinite(listScore)
            ? listScore
            : typeof analyzedScore === 'number' && Number.isFinite(analyzedScore)
              ? analyzedScore
              : prev.sentimentScore;

    const sentiment =
        listArticle.sentiment ??
        analysis.sentiment ??
        polarityFromUnitScore(
            typeof sentimentScore === 'number' && Number.isFinite(sentimentScore) ? sentimentScore : undefined,
        );

    return {
        ...prev,
        detailedSummary:
            analysis.detailedSummary ??
            prev.detailedSummary ??
            listArticle.detailedSummary ??
            undefined,
        summary:
            listArticle.summary?.trim()
                ? listArticle.summary
                : (analysis.summary ?? prev.summary ?? listArticle.summary),
        keyTakeaways: analysis.keyTakeaways ?? prev.keyTakeaways ?? listArticle.keyTakeaways,
        sentimentScore,
        sentiment,
    };
}

const ArticleDetailModal: React.FC<ArticleDetailModalProps> = ({ article, onClose }) => {
    const { isAuthenticated } = useAuth();
    const location = useLocation();
    const [enrichedArticle, setEnrichedArticle] = useState<NewsArticle>(article);
    const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>(AnalysisStatus.IDLE);

    // Q&A State
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState<string | null>(null);
    const [isThinking, setIsThinking] = useState(false);

    useEffect(() => {
        let cancelled = false;
        setEnrichedArticle(article);
        setAnswer(null);

        const wantsAi =
            !article.detailedSummary?.trim() ||
            !article.keyTakeaways ||
            article.keyTakeaways.length === 0;

        const loadAnalysis = async () => {
            if (!isAuthenticated) {
                setAnalysisStatus(AnalysisStatus.IDLE);
                return;
            }
            if (!wantsAi) {
                setAnalysisStatus(AnalysisStatus.SUCCESS);
                return;
            }
            setAnalysisStatus(AnalysisStatus.LOADING);
            try {
                const analysis = await analyzeArticle(article);
                if (cancelled) {
                    return;
                }
                setEnrichedArticle((prev) => mergeArticleAnalysis(prev, article, analysis));
                setAnalysisStatus(AnalysisStatus.SUCCESS);
            } catch {
                if (!cancelled) {
                    setAnalysisStatus(AnalysisStatus.ERROR);
                }
            }
        };
        void loadAnalysis();
        return () => {
            cancelled = true;
        };
    }, [article, isAuthenticated]);

    const effectiveSentiment =
        enrichedArticle.sentiment ??
        polarityFromUnitScore(
            typeof enrichedArticle.sentimentScore === 'number' ? enrichedArticle.sentimentScore : undefined,
        );

    const handleQuery = async () => {
        if (!isAuthenticated) return;
        if (!question.trim()) return;
        setIsThinking(true);
        setAnswer(null);
        try {
            const context = `
            Title: ${enrichedArticle.title}
            Source: ${enrichedArticle.source}
            Snippet: ${enrichedArticle.snippet}
            Summary: ${enrichedArticle.detailedSummary || enrichedArticle.summary}
            Key Takeaways: ${enrichedArticle.keyTakeaways?.join(', ')}
        `;
            const result = await askNewsContext(context, question);
            setAnswer(result);
        } catch {
            setAnswer('Unable to answer questions about this article.');
        } finally {
            setIsThinking(false);
        }
    };

    return createPortal(
        <div className="fixed inset-0 z-[100] flex min-h-[100dvh] w-full items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm animate-in fade-in duration-200 dark:bg-black/65">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl relative animate-in zoom-in-95 duration-200">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300 transition-colors z-10"
                >
                    <X size={20} />
                </button>

                {/* Header Image/Gradient */}
                <div className="h-24 bg-gradient-to-r from-indigo-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 w-full shrink-0 relative border-b border-slate-200 dark:border-slate-700">
                    <div className="absolute -bottom-6 left-6 w-12 h-12 rounded-xl bg-white dark:bg-slate-950 border-2 border-slate-200 dark:border-slate-600 shadow-md overflow-hidden flex items-stretch">
                        <NewsSourceLogo
                            source={enrichedArticle.source}
                            articleUrl={enrichedArticle.url}
                            iconSize={24}
                            fillSquare
                        />
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 pt-8 overflow-y-auto custom-scrollbar flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/50 px-2 py-1 rounded">
                            {enrichedArticle.source}
                        </span>
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                            {formatDateToLocalWithOffset(enrichedArticle.timestamp)}
                        </span>
                    </div>

                    <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4 leading-snug">{enrichedArticle.title}</h2>

                    {!isAuthenticated && enrichedArticle.summary?.trim() ? (
                        <p className="text-slate-600 dark:text-slate-300 text-sm mb-6 leading-relaxed">{enrichedArticle.summary.trim()}</p>
                    ) : null}

                    {/* AI Section */}
                    <div className="border-t border-slate-100 dark:border-slate-800 pt-6">
                        <div className="flex items-center gap-2 mb-4">
                            <BrainCircuit className="text-indigo-600 dark:text-indigo-400" />
                            <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Sibyl Analysis</h3>
                        </div>

                        {!isAuthenticated ? (
                            <div className="flex flex-col items-center justify-center py-10 text-center text-slate-500 dark:text-slate-400">
                                <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Login to unlock AI summary</p>
                                <p className="text-xs mt-1 opacity-90">AI insights are available after signing in.</p>
                                <NavLink
                                    to="/login"
                                    state={{ from: `/news?articleId=${article.id}` || location.pathname }}
                                    className="mt-4 px-3 py-2 rounded-lg bg-indigo-600 dark:bg-indigo-500 text-white text-xs font-bold hover:bg-indigo-700 dark:hover:bg-indigo-400"
                                >
                                    Login to Continue
                                </NavLink>
                            </div>
                        ) : analysisStatus === AnalysisStatus.LOADING ? (
                            <div className="flex flex-col items-center justify-center py-12 text-slate-500 dark:text-slate-400">
                                <Loader2 className="animate-spin mb-2 text-indigo-600 dark:text-indigo-400" size={32} />
                                <p>Generating detailed report...</p>
                            </div>
                        ) : (
                            <div className="space-y-6">
                                {/* Detailed Summary */}
                                <div>
                                    <h4 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                                        Executive Summary
                                    </h4>
                                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed text-sm">
                                        {enrichedArticle.detailedSummary ||
                                            enrichedArticle.summary?.trim() ||
                                            'Summary unavailable.'}
                                    </p>
                                </div>

                                {/* Key Takeaways */}
                                {enrichedArticle.keyTakeaways && enrichedArticle.keyTakeaways.length > 0 ? (
                                    <div>
                                        <h4 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                                            Key Takeaways
                                        </h4>
                                        <ul className="space-y-2">
                                            {enrichedArticle.keyTakeaways.map((point, idx) => (
                                                <li key={idx} className="flex items-start gap-2 text-sm text-slate-700 dark:text-slate-300">
                                                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-600 dark:bg-indigo-400 shrink-0"></span>
                                                    <span>{point}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                ) : null}

                                {/* Stats Row — list/Supabase score takes priority over ephemeral analyze sentiment */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-slate-50 dark:bg-slate-800/50 p-3 rounded-xl border border-slate-200 dark:border-slate-700">
                                        <span className="text-xs text-slate-500 dark:text-slate-400 uppercase">Sentiment score</span>
                                        <div className="flex items-center gap-2 mt-1">
                                            <div className="h-2 w-full bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
                                                <div
                                                    className={
                                                        effectiveSentiment === 'Positive'
                                                            ? 'h-full bg-emerald-500'
                                                            : effectiveSentiment === 'Negative'
                                                              ? 'h-full bg-red-500'
                                                              : 'h-full bg-amber-400'
                                                    }
                                                    style={{
                                                        width: `${Math.max(
                                                            0,
                                                            Math.min(
                                                                100,
                                                                ((Math.max(-1, Math.min(1, enrichedArticle.sentimentScore ?? 0)) +
                                                                    1) /
                                                                    2) *
                                                                    100,
                                                            ),
                                                        )}%`,
                                                    }}
                                                />
                                            </div>
                                            <span className="font-bold text-slate-900 dark:text-slate-100 whitespace-nowrap tabular-nums">
                                                {formatUnitSentiment(enrichedArticle.sentimentScore)}
                                            </span>
                                        </div>
                                        <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">Scale −1 bearish … +1 bullish</p>
                                    </div>
                                    <div className="bg-slate-50 dark:bg-slate-800/50 p-3 rounded-xl border border-slate-200 dark:border-slate-700">
                                        <span className="text-xs text-slate-500 dark:text-slate-400 uppercase">Sentiment</span>
                                        <div
                                            className={`font-bold mt-1 ${
                                                effectiveSentiment === 'Positive'
                                                    ? 'text-green-600 dark:text-emerald-400'
                                                    : effectiveSentiment === 'Negative'
                                                      ? 'text-red-600 dark:text-rose-400'
                                                      : 'text-slate-500 dark:text-slate-400'
                                            }`}
                                        >
                                            {effectiveSentiment}
                                        </div>
                                    </div>
                                </div>

                                {/* Embedded Article Chat */}
                                <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800">
                                    <h4 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <Sparkles size={14} className="text-indigo-600 dark:text-indigo-400" /> Ask about this article
                                    </h4>

                                    {answer && (
                                        <div className="bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-100 dark:border-indigo-900/50 rounded-lg p-3 mb-3 text-sm text-slate-700 dark:text-slate-200">
                                            {answer}
                                        </div>
                                    )}

                                    <div className="relative">
                                        <input
                                            type="text"
                                            value={question}
                                            onChange={(e) => setQuestion(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                                            placeholder="e.g. Why is this important?"
                                            disabled={!isAuthenticated}
                                            className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-600 rounded-lg py-2.5 pl-3 pr-10 text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:border-indigo-500 dark:focus:border-indigo-400"
                                        />
                                        <button
                                            onClick={handleQuery}
                                            disabled={!isAuthenticated || isThinking || !question.trim()}
                                            className="absolute right-1.5 top-1.5 p-1.5 bg-indigo-600 dark:bg-indigo-500 rounded-md text-white disabled:opacity-50 hover:bg-indigo-700 dark:hover:bg-indigo-400"
                                        >
                                            {isThinking ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                                        </button>
                                    </div>
                                </div>

                            </div>
                        )}
                    </div>
                </div>

                {/* Footer Action */}
                <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/80">
                    <a
                        href={enrichedArticle.url || '#'}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center justify-center gap-2 w-full py-3 bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-400 text-white font-medium rounded-xl transition-colors"
                    >
                        Read Full Article <ExternalLink size={16} />
                    </a>
                </div>
            </div>
        </div>,
        document.body,
    );
};

export default ArticleDetailModal;
