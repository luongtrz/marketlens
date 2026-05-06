import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { refreshAuth, signIn, signUp } from '../services/apiService';

const STORAGE_KEY = 'marketlens_auth';

type AuthState = {
  isAuthenticated: boolean;
  email: string | null;
  accessToken: string | null;
  refreshToken: string | null;
  expiresAt: number | null;
};

type AuthContextValue = AuthState & {
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  refresh: () => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const readStoredAuth = (): AuthState => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { isAuthenticated: false, email: null, accessToken: null, refreshToken: null, expiresAt: null };
    const parsed = JSON.parse(raw) as {
      email?: string;
      accessToken?: string;
      refreshToken?: string;
      expiresAt?: number;
    };
    const isValid = Boolean(parsed?.accessToken && parsed.expiresAt && Date.now() < parsed.expiresAt);
    return {
      isAuthenticated: isValid,
      email: parsed.email || null,
      accessToken: parsed.accessToken || null,
      refreshToken: parsed.refreshToken || null,
      expiresAt: parsed.expiresAt || null,
    };
  } catch {
    return { isAuthenticated: false, email: null, accessToken: null, refreshToken: null, expiresAt: null };
  }
};

const writeStoredAuth = (payload: {
  email: string;
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
};

const clearStoredAuth = () => {
  localStorage.removeItem(STORAGE_KEY);
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AuthState>(() => readStoredAuth());

  const applyAuth = (data: { access_token: string; refresh_token: string; expires_in: number; user?: { email?: string } }, fallbackEmail: string) => {
    const expiresAt = Date.now() + data.expires_in * 1000;
    const email = data.user?.email || fallbackEmail;
    writeStoredAuth({
      email,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt,
    });
    setState({
      isAuthenticated: true,
      email,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt,
    });
  };

  const login = async (email: string, password: string) => {
    const trimmedEmail = email.trim();
    if (!trimmedEmail || !password.trim()) {
      throw new Error('Email and password are required.');
    }
    if (new TextEncoder().encode(password).length > 72) {
      throw new Error('Password must be 72 bytes or fewer.');
    }
    const data = await signIn(trimmedEmail, password);
    applyAuth(data, trimmedEmail);
  };

  const signup = async (email: string, password: string) => {
    const trimmedEmail = email.trim();
    if (!trimmedEmail || !password.trim()) {
      throw new Error('Email and password are required.');
    }
    if (new TextEncoder().encode(password).length > 72) {
      throw new Error('Password must be 72 bytes or fewer.');
    }
    const data = await signUp(trimmedEmail, password);
    applyAuth(data, trimmedEmail);
  };

  const refresh = async () => {
    if (!state.refreshToken) {
      throw new Error('No refresh token available.');
    }
    const data = await refreshAuth(state.refreshToken);
    applyAuth(data, state.email || 'user@marketlens.ai');
  };

  const logout = () => {
    clearStoredAuth();
    setState({ isAuthenticated: false, email: null, accessToken: null, refreshToken: null, expiresAt: null });
  };

  useEffect(() => {
    if (!state.refreshToken || !state.expiresAt) return;
    if (Date.now() < state.expiresAt - 30_000) return;

    let cancelled = false;
    const doRefresh = async () => {
      try {
        const data = await refreshAuth(state.refreshToken as string);
        if (!cancelled) {
          applyAuth(data, state.email || 'user@marketlens.ai');
        }
      } catch {
        if (!cancelled) logout();
      }
    };

    void doRefresh();
    return () => {
      cancelled = true;
    };
  }, [state.refreshToken, state.expiresAt]);

  const value = useMemo<AuthContextValue>(
    () => ({
      ...state,
      login,
      signup,
      refresh,
      logout,
    }),
    [state]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider.');
  }
  return ctx;
};
