/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: ['class'],
    content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        main: '#f8fafc',
        surface: '#ffffff',
        base: '#0f172a',
        muted: '#475569',
        subtle: '#94a3b8',
        accent: {
          DEFAULT: '#2563eb',
          hover: '#1d4ed8',
        },
        success: '#22c55e',
        error: '#ef4444',
        border: '#e2e8f0',
      },
      boxShadow: {
        'saas': '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)',
        'saas-hover': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
      }
    },
  },
  plugins: [require("tailwindcss-animate")],
}
