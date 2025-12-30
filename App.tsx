import React from 'react';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import NewsAnalysis from './pages/NewsAnalysis';

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans flex flex-col transition-colors duration-200">
        <Navbar />
        
        <main className="flex-1 pt-14 overflow-hidden">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/news" element={<NewsAnalysis />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;