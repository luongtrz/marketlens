import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { NewsArticle, AnalysisStatus } from '../types';
import { analyzeArticle, askNewsContext } from '../services/apiService';
import { Loader2, X, BrainCircuit, Sparkles, Send, Globe, ExternalLink } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { formatDateToLocalWithOffset } from '../utils/formatters';

interface ArticleDetailModalProps {
    article: NewsArticle;
    onClose: () => void;
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
        const loadAnalysis = async () => {
            if (!isAuthenticated) {
                setAnalysisStatus(AnalysisStatus.IDLE);
                return;
            }
            if (!article.detailedSummary || !article.keyTakeaways) {
                setAnalysisStatus(AnalysisStatus.LOADING);
                try {
                    const analysis = await analyzeArticle(article);
                    setEnrichedArticle(prev => ({ ...prev, ...analysis }));
                    setAnalysisStatus(AnalysisStatus.SUCCESS);
                } catch (e) {
                    setAnalysisStatus(AnalysisStatus.ERROR);
                }
            } else {
                setAnalysisStatus(AnalysisStatus.SUCCESS);
            }
        };
        loadAnalysis();
    }, [article]);

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
        } catch (e) {
            setAnswer("Unable to answer questions about this article.");
        } finally {
            setIsThinking(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white border border-slate-200 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl relative animate-in zoom-in-95 duration-200">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors z-10"
                >
                    <X size={20} />
                </button>

                {/* Header Image/Gradient */}
                <div className="h-24 bg-gradient-to-r from-indigo-50 to-slate-100 w-full shrink-0 relative border-b border-slate-200">
                    <div className="absolute -bottom-6 left-6 w-12 h-12 rounded-xl bg-white border-2 border-slate-200 flex items-center justify-center text-indigo-600 shadow-md">
                        <Globe size={24} />
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 pt-8 overflow-y-auto custom-scrollbar flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-1 rounded">{enrichedArticle.source}</span>
                        <span className="text-xs text-slate-500">{formatDateToLocalWithOffset(enrichedArticle.timestamp)}</span>
                    </div>

                    <h2 className="text-xl font-bold text-slate-900 mb-4 leading-snug">{enrichedArticle.title}</h2>

                    <div className="text-slate-600 text-sm mb-6 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100">
                        {enrichedArticle.snippet}
                    </div>

                    {/* AI Section */}
                    <div className="border-t border-slate-100 pt-6">
                        <div className="flex items-center gap-2 mb-4">
                            <BrainCircuit className="text-indigo-600" />
                            <h3 className="text-lg font-bold text-slate-900">Sibyl Analysis</h3>
                        </div>

                        {!isAuthenticated ? (
                            <div className="flex flex-col items-center justify-center py-10 text-center text-slate-500">
                                <p className="text-sm font-semibold text-slate-700">Login to unlock AI summary</p>
                                <p className="text-xs mt-1">AI insights are available after signing in.</p>
                                <NavLink
                                    to="/login"
                                    state={{ from: `/news?articleId=${article.id}` || location.pathname }}
                                    className="mt-4 px-3 py-2 rounded-lg bg-indigo-600 text-white text-xs font-bold hover:bg-indigo-700"
                                >
                                    Login to Continue
                                </NavLink>
                            </div>
                        ) : analysisStatus === AnalysisStatus.LOADING ? (
                            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                                <Loader2 className="animate-spin mb-2 text-indigo-600" size={32} />
                                <p>Generating detailed report...</p>
                            </div>
                        ) : (
                            <div className="space-y-6">
                                {/* Detailed Summary */}
                                <div>
                                    <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Executive Summary</h4>
                                    <p className="text-slate-700 leading-relaxed text-sm">
                                        {enrichedArticle.detailedSummary || enrichedArticle.summary || "Summary unavailable."}
                                    </p>
                                </div>

                                {/* Key Takeaways */}
                                {enrichedArticle.keyTakeaways && (
                                    <div>
                                        <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">Key Takeaways</h4>
                                        <ul className="space-y-2">
                                            {enrichedArticle.keyTakeaways.map((point, idx) => (
                                                <li key={idx} className="flex items-start gap-2 text-sm text-slate-700">
                                                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-600 shrink-0"></span>
                                                    <span>{point}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Stats Row */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                                        <span className="text-xs text-slate-500 uppercase">Market Impact</span>
                                        <div className="flex items-center gap-2 mt-1">
                                            <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                                                <div className="h-full bg-orange-500" style={{ width: `${enrichedArticle.impactScore || 0}%` }}></div>
                                            </div>
                                            <span className="font-bold text-slate-900">{enrichedArticle.impactScore}/100</span>
                                        </div>
                                    </div>
                                    <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                                        <span className="text-xs text-slate-500 uppercase">Sentiment</span>
                                        <div className={`font-bold mt-1 ${enrichedArticle.sentiment === 'Positive' ? 'text-green-600' :
                                                enrichedArticle.sentiment === 'Negative' ? 'text-red-600' : 'text-slate-500'
                                            }`}>
                                            {enrichedArticle.sentiment || 'Neutral'}
                                        </div>
                                    </div>
                                </div>

                                {/* Embedded Article Chat */}
                                <div className="mt-6 pt-6 border-t border-slate-100">
                                    <h4 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <Sparkles size={14} className="text-indigo-600" /> Ask about this article
                                    </h4>

                                    {answer && (
                                        <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3 mb-3 text-sm text-slate-700">
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
                                            className="w-full bg-white border border-slate-200 rounded-lg py-2.5 pl-3 pr-10 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
                                        />
                                        <button
                                            onClick={handleQuery}
                                            disabled={!isAuthenticated || isThinking || !question.trim()}
                                            className="absolute right-1.5 top-1.5 p-1.5 bg-indigo-600 rounded-md text-white disabled:opacity-50 hover:bg-indigo-700"
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
                <div className="p-4 border-t border-slate-200 bg-slate-50">
                    <a
                        href={enrichedArticle.url || '#'}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center justify-center gap-2 w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition-colors"
                    >
                        Read Full Article <ExternalLink size={16} />
                    </a>
                </div>
            </div>
        </div>
    );
};

export default ArticleDetailModal;