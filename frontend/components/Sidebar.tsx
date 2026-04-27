import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Newspaper, MessageSquareText, Activity, Settings, TrendingUp } from 'lucide-react';

const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', icon: <LayoutDashboard size={20} />, label: 'Dashboard' },
    { path: '/news', icon: <Newspaper size={20} />, label: 'News Intelligence' },
    { path: '/chat', icon: <MessageSquareText size={20} />, label: 'AI Analyst' },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-slate-900 border-r border-slate-800 hidden md:flex flex-col">
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-2 text-indigo-500">
          <Activity size={28} />
          <h1 className="text-xl font-bold text-white tracking-tight">CryptoSibyl</h1>
        </div>
        <p className="text-xs text-slate-500 mt-1">AI-Powered Market Forecaster</p>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                isActive
                  ? 'bg-indigo-600/10 text-indigo-400 font-medium border border-indigo-600/20'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`
            }
          >
            {item.icon}
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-2 mb-2 text-emerald-400">
            <TrendingUp size={16} />
            <span className="text-xs font-bold uppercase">Market Status</span>
          </div>
          <p className="text-sm text-slate-300 font-medium">Accumulation Phase</p>
          <div className="w-full bg-slate-700 h-1 mt-2 rounded-full overflow-hidden">
            <div className="bg-emerald-500 w-[65%] h-full"></div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;