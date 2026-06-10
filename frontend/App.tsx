import React from 'react';
import { HashRouter as Router, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import NewsAnalysis from './pages/NewsAnalysis';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import { AuthProvider } from './context/AuthContext';

const AppContent: React.FC = () => {
  const location = useLocation();
  const isDashboardRoute = location.pathname === '/';
  const isNewsRoute = location.pathname === '/news';
  const isWorkspaceRoute = isDashboardRoute || isNewsRoute;

  if (!isWorkspaceRoute) {
    return (
      <main className="flex flex-1 min-h-0 flex-col pt-14">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    );
  }

  return (
    <main className="flex flex-1 min-h-0 flex-col pt-14">
      <div
        className={`${isDashboardRoute ? 'flex' : 'hidden'} min-h-0 flex-1 flex-col overflow-hidden`}
        aria-hidden={!isDashboardRoute}
      >
        <Dashboard />
      </div>
      <div
        className={`${isNewsRoute ? 'flex' : 'hidden'} min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain`}
        aria-hidden={!isNewsRoute}
      >
        <NewsAnalysis />
      </div>
    </main>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <Router>
        <div className="h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans flex flex-col transition-colors duration-200">
          <Navbar />
          <AppContent />
        </div>
      </Router>
    </AuthProvider>
  );
};

export default App;
