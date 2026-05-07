import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setSent(true);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 dark:bg-slate-950">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-8 shadow-lg dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-6 text-center">
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Forgot password</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">We will send reset instructions to your email.</p>
        </div>

        {sent ? (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-300">
            If this email exists, reset instructions have been sent.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" htmlFor="email">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-indigo-400 dark:focus:ring-indigo-400"
                placeholder="you@example.com"
                required
              />
            </div>

            <button
              type="submit"
              className="w-full rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-400"
            >
              Send reset link
            </button>
          </form>
        )}

        <div className="mt-4 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
          Back to
          <NavLink
            to="/login"
            className="ml-1 font-semibold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
          >
            Login
          </NavLink>
        </div>

        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-400">
          Demo only. Email is not sent yet.
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
