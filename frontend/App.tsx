import React from 'react';
import { HashRouter as Router, Navigate, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import NewsAnalysis from './pages/NewsAnalysis';

const AppContent: React.FC = () => {
  const location = useLocation();
  const isDashboard = location.pathname === '/';
  const isNews = location.pathname === '/news';

  if (!isDashboard && !isNews) {
    return <Navigate to="/" replace />;
  }

  return (
    <main className="flex flex-1 min-h-0 flex-col pt-14">
      <div className={`min-h-0 flex-1 flex-col overflow-hidden ${isDashboard ? 'flex' : 'hidden'}`}>
        <Dashboard />
      </div>
      <div className={`min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain ${isNews ? 'flex' : 'hidden'}`}>
        <NewsAnalysis />
      </div>
    </main>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <div className="h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans flex flex-col transition-colors duration-200">
        <Navbar />
        <AppContent />
      </div>
    </Router>
  );
};

export default App;