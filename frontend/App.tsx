import React from 'react';
import { HashRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import NewsAnalysis from './pages/NewsAnalysis';
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import { AuthProvider } from './context/AuthContext';

const AppContent: React.FC = () => (
  <main className="flex flex-1 min-h-0 flex-col pt-14">
    <Routes>
      <Route
        path="/"
        element={
          <div className="min-h-0 flex-1 flex-col overflow-hidden flex">
            <Dashboard />
          </div>
        }
      />
      <Route
        path="/news"
        element={
          <div className="min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain flex">
            <NewsAnalysis />
          </div>
        }
      />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </main>
);

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