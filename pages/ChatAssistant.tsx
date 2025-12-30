import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { createChatSession } from '../services/geminiService';
import { Send, Bot, User, Loader2, Info, ExternalLink, Globe } from 'lucide-react';

const ChatAssistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'model',
      content: 'Hello! I am your AI Market Analyst. I can answer questions about crypto trends, technical indicators, or recent news events. What would you like to know?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatSession = useRef(createChatSession());
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const result = await chatSession.current.sendMessage(userMsg.content);
      const response = result.response;
      const responseText = response.text; 

      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        content: responseText || "I have processed your request.",
        timestamp: new Date(),
        groundingMetadata: response.candidates?.[0]?.groundingMetadata
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat error", error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        content: "I'm having trouble connecting to the market data right now. Please try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderGroundingSources = (metadata: ChatMessage['groundingMetadata']) => {
    if (!metadata || !metadata.groundingChunks || metadata.groundingChunks.length === 0) return null;

    const uniqueSources = new Map();
    metadata.groundingChunks.forEach(chunk => {
      if (chunk.web) {
        uniqueSources.set(chunk.web.uri, chunk.web.title);
      }
    });

    if (uniqueSources.size === 0) return null;

    return (
      <div className="mt-3 pt-2 border-t border-slate-200">
        <div className="flex items-center gap-1.5 text-[10px] text-slate-500 uppercase font-bold mb-2">
          <Globe size={10} /> Sources
        </div>
        <div className="flex flex-wrap gap-2">
          {Array.from(uniqueSources.entries()).map(([uri, title], idx) => (
            <a 
              key={idx}
              href={uri}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 bg-white hover:bg-slate-50 text-indigo-600 text-xs px-2 py-1.5 rounded-md transition-colors border border-slate-200 hover:border-indigo-200 truncate max-w-[200px]"
            >
              <span className="truncate">{title}</span>
              <ExternalLink size={10} className="shrink-0 opacity-50" />
            </a>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="h-[calc(100vh-2rem)] md:h-[calc(100vh-3rem)] flex flex-col bg-white rounded-2xl border border-slate-200 overflow-hidden m-4 shadow-sm">
        {/* Chat Header */}
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <Bot className="text-white" size={24} />
                </div>
                <div>
                    <h2 className="font-bold text-slate-900">Sibyl Assistant</h2>
                    <p className="text-xs text-indigo-600 flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Online
                    </p>
                </div>
            </div>
            <div className="hidden md:block">
                <span className="text-xs bg-white text-slate-500 px-3 py-1 rounded-full border border-slate-200 flex items-center gap-1 shadow-sm">
                   <Globe size={10} /> Google Search Grounding
                </span>
            </div>
        </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50/50" ref={scrollRef}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[90%] md:max-w-[75%] rounded-2xl px-5 py-4 ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg shadow-indigo-500/10'
                  : 'bg-white text-slate-800 rounded-tl-none shadow-sm border border-slate-100'
              }`}
            >
              <div className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</div>
              
              {/* Render Search Sources */}
              {msg.role === 'model' && renderGroundingSources(msg.groundingMetadata)}

              <div className={`text-[10px] mt-2 opacity-50 ${msg.role === 'user' ? 'text-indigo-100' : 'text-slate-400'}`}>
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl rounded-tl-none px-5 py-3 flex items-center gap-2 shadow-sm border border-slate-100">
              <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
              <span className="text-xs text-slate-500">Sibyl is researching...</span>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-slate-200">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about market trends, news, or specific coins..."
            className="w-full bg-slate-50 text-slate-900 border border-slate-200 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all placeholder-slate-400"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="absolute right-2 top-2 p-1.5 bg-indigo-600 rounded-lg text-white disabled:opacity-50 hover:bg-indigo-700 transition"
          >
            <Send size={18} />
          </button>
        </div>
        <div className="mt-2 flex items-center justify-center gap-1 text-[10px] text-slate-400">
            <Info size={10} />
            <span>AI responses are grounded in Google Search but may vary.</span>
        </div>
      </div>
    </div>
  );
};

export default ChatAssistant;