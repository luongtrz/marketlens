import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Signup: React.FC = () => {
  const navigate = useNavigate();
  const { signup } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setIsSubmitting(true);

    try {
      await signup(email, password);
      navigate('/', { replace: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Sign up failed.';
      setError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 dark:bg-slate-950">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-8 shadow-lg dark:border-slate-700 dark:bg-slate-900">
        <div className="mb-6 text-center">
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Create your account</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">Start using MarketLens in seconds.</p>
        </div>

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

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" htmlFor="password">
              Password
            </label>
            <div className="relative mt-1">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="new-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 pr-20 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-indigo-400 dark:focus:ring-indigo-400"
                placeholder="••••••••"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300" htmlFor="confirmPassword">
              Confirm password
            </label>
            <div className="relative mt-1">
              <input
                id="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 pr-20 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-indigo-400 dark:focus:ring-indigo-400"
                placeholder="••••••••"
                required
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword((prev) => !prev)}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
              >
                {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-indigo-500 dark:hover:bg-indigo-400"
          >
            {isSubmitting ? 'Creating account...' : 'Sign up'}
          </button>
        </form>

        <div className="mt-4 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
          Already have an account?
          <NavLink
            to="/login"
            className="ml-1 font-semibold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
          >
            Login
          </NavLink>
        </div>

        <div className="mt-3 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
          Forgot your password?
          <NavLink
            to="/forgot-password"
            className="ml-1 font-semibold text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
          >
            Reset it
          </NavLink>
        </div>

        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-400">
          Demo signup only. Any email/password will work.
        </div>
      </div>
    </div>
  );
};

export default Signup;
