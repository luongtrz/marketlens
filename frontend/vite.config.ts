import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: env.MAIN_CONTROLLER_PUBLIC_URL || 'http://127.0.0.1:8005',
          changeOrigin: true,
        },
        '/market': {
          target: env.MARKET_DATA_PUBLIC_URL || 'http://127.0.0.1:8002',
          changeOrigin: true,
          ws: true,
          rewrite: (path) => path.replace(/^\/market/, ''),
        },
      },
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.COINGECKO_API_KEY': JSON.stringify(env.COINGECKO_API_KEY),
      'process.env.CRYPTOCOMPARE_API_KEY': JSON.stringify(env.CRYPTOCOMPARE_API_KEY)
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    }
  };
});
