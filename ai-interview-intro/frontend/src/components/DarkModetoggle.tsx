import { useEffect, useState } from 'react';

export default function DarkModeToggle() {
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    const stored = localStorage.getItem('theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  return (
    <button
      id="dark-mode-toggle"
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      onClick={() => setIsDark((prev) => !prev)}
      className={`
        relative w-16 h-8 flex items-center rounded-full p-1
        transition-all duration-500 ease-in-out cursor-pointer
        hover:scale-105 active:scale-95
        ${isDark
          ? 'bg-gradient-to-r from-gray-800 to-gray-900'
          : 'bg-gradient-to-r from-yellow-300 to-orange-400'
        }
      `}
    >
      {/* Glow Aura */}
      <div
        className={`
          absolute inset-0 rounded-full blur-lg opacity-40 transition-all duration-500
          ${isDark ? 'bg-blue-500' : 'bg-yellow-400'}
        `}
      />

      {/* Sliding knob */}
      <div
        className={`
          relative z-10 w-6 h-6 rounded-full flex items-center justify-center
          transition-all duration-500 ease-in-out
          ${isDark
            ? 'translate-x-8 bg-black text-yellow-300 shadow-[0_0_14px_rgba(255,255,150,0.7)]'
            : 'translate-x-0 bg-white dark:bg-black text-yellow-500 shadow-[0_0_12px_rgba(255,200,0,0.6)]'
          }
        `}
      >
        {/* 🌙 Moon */}
        <svg
          className={`absolute w-4 h-4 transition-all duration-500 ${
            isDark ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
          }`}
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 
          7 7 0 0 0 21 12.79z" />
        </svg>

        {/* ☀️ Sun (FIXED with rays) */}
        <svg
          className={`absolute w-4 h-4 transition-all duration-500 ${
            isDark ? 'opacity-0 scale-50' : 'opacity-100 scale-100'
          }`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        >
          <circle cx="12" cy="12" r="4" fill="currentColor" />

          {/* Rays */}
          <line x1="12" y1="1" x2="12" y2="4" />
          <line x1="12" y1="20" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="6" y2="6" />
          <line x1="18" y1="18" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="4" y2="12" />
          <line x1="20" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="6" y2="18" />
          <line x1="18" y1="6" x2="19.78" y2="4.22" />
        </svg>
      </div>

      {/* Background icons */}
      <div className="absolute inset-0 flex justify-between items-center px-2 text-xs pointer-events-none">
        <span className={`transition-opacity duration-500 ${isDark ? 'opacity-0' : 'opacity-100'}`}>
          ☀️
        </span>
        <span className={`transition-opacity duration-500 ${isDark ? 'opacity-100' : 'opacity-0'}`}>
          🌙
        </span>
      </div>
    </button>
  );
}