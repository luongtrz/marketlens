import React, { useState, useRef, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Newspaper, ScanEye, ChevronDown, User, Settings, CreditCard, LogOut, Shield, Moon, Sun } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar: React.FC = () => {
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
    const { isAuthenticated, email, logout } = useAuth();
  
  // Theme State
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  

  // Initialize Theme based on system or storage (mocked here as default light)
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };


  const navItems = [
    { path: '/', icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
    { path: '/news', icon: <Newspaper size={18} />, label: 'News Intelligence' },
  ];

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setShowProfileMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <nav className="fixed top-0 left-0 w-full h-14 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-4 z-50 shadow-sm transition-colors duration-200">
        {/* Left: Logo & Nav */}
        <div className="flex items-center gap-8">
            {/* Logo */}
            <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400">
                <ScanEye size={24} strokeWidth={2.5} />
                <h1 className="text-lg font-bold text-slate-900 dark:text-white tracking-tight">MarketLens</h1>
            </div>

            {/* Navigation Links */}
            <div className="hidden md:flex items-center gap-1">
                {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                    `flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all ${
                        isActive
                        ? 'bg-slate-100 dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 font-medium'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                    }`
                    }
                >
                    {item.icon}
                    <span>{item.label}</span>
                </NavLink>
                ))}
            </div>
        </div>

        {/* Center/Right: Controls & Account */}
        <div className="flex items-center gap-3">
            
            {/* Theme Toggle */}
            <button 
              onClick={toggleTheme}
              className="p-2 rounded-lg text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all"
              title="Toggle Theme"
            >
                {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
            </button>

            <div className="h-6 w-px bg-slate-200 dark:bg-slate-700 mx-1"></div>

            {/* Account Section */}
            {isAuthenticated ? (
                <div className="relative" ref={profileRef}>
                    <button 
                        onClick={() => setShowProfileMenu(!showProfileMenu)}
                        className={`flex items-center gap-2 pl-1 pr-2 py-1 rounded-full border transition-all ${showProfileMenu ? 'bg-indigo-50 dark:bg-slate-800 border-indigo-200 dark:border-indigo-900 ring-2 ring-indigo-500/10' : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800'}`}
                    >
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white shadow-sm border-2 border-white dark:border-slate-800">
                            {(email || 'User').slice(0, 2).toUpperCase()}
                        </div>
                        <ChevronDown size={14} className={`text-slate-400 transition-transform ${showProfileMenu ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Dropdown Menu */}
                    {showProfileMenu && (
                        <div className="absolute right-0 top-full mt-2 w-64 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xl py-1 animate-in fade-in zoom-in-95 duration-200 origin-top-right">
                            
                            {/* Profile Header */}
                            <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800">
                                <div className="flex items-center justify-between mb-1">
                                    <h3 className="font-bold text-sm text-slate-900 dark:text-white">MarketLens User</h3>
                                    <span className="text-[10px] bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 px-1.5 py-0.5 rounded font-bold border border-indigo-200 dark:border-indigo-800">BETA</span>
                                </div>
                                <p className="text-xs text-slate-500 dark:text-slate-400 truncate">{email || 'user@marketlens.ai'}</p>
                            </div>

                            {/* Menu Items */}
                            <div className="py-1">
                                <button className="w-full text-left px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-2.5 transition-colors">
                                    <User size={16} className="text-slate-400" />
                                    My Profile
                                </button>
                                <button className="w-full text-left px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-2.5 transition-colors">
                                    <CreditCard size={16} className="text-slate-400" />
                                    Subscription & Billing
                                </button>
                                <button className="w-full text-left px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-2.5 transition-colors">
                                    <Shield size={16} className="text-slate-400" />
                                    API Keys
                                </button>
                                <button className="w-full text-left px-4 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-2.5 transition-colors">
                                    <Settings size={16} className="text-slate-400" />
                                    Settings
                                </button>
                            </div>

                            {/* Logout */}
                            <div className="border-t border-slate-100 dark:border-slate-800 py-1">
                                <button
                                    onClick={() => {
                                      logout();
                                      setShowProfileMenu(false);
                                    }}
                                    className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/10 flex items-center gap-2.5 transition-colors"
                                >
                                    <LogOut size={16} />
                                    Sign Out
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            ) : (
              <div className="flex items-center gap-2">
                <NavLink
                  to="/signup"
                  className="px-3 py-1.5 rounded-lg text-xs font-bold border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-100 dark:hover:bg-indigo-900/40 transition-all"
                >
                  Sign Up
                </NavLink>
                <NavLink
                  to="/login"
                  className="px-3 py-1.5 rounded-lg text-xs font-bold border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-all"
                >
                  Login
                </NavLink>
              </div>
            )}
        </div>
    </nav>
  );
};

export default Navbar;