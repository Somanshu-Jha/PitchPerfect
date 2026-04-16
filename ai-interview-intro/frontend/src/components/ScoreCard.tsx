import { useEffect, useState } from 'react';

interface ScoreCardProps {
  score: number; // 0–100
}

/**
 * ScoreCard – Clean minimal numeric display with smooth 1s JS-driven animation.
 */
export default function ScoreCard({ score }: ScoreCardProps) {
  const [displayed, setDisplayed] = useState(0);

  // Smooth count-up animation 1s using requestAnimationFrame
  useEffect(() => {
    let startTimestamp: number | null = null;
    const duration = 1000; // 1s

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // easeOutCubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayed(Math.round(eased * score));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }, [score]);

  const colorClass =
    score >= 80 ? 'text-success bg-green-50' :
    score >= 60 ? 'text-orange-500 bg-orange-50' :
    'text-error bg-red-50';

  const label =
    score >= 80 ? 'Excellent' :
    score >= 60 ? 'Good'      :
    'Needs Work';

  return (
    <div className="saas-card p-10 flex flex-col justify-between items-start h-full min-h-[260px] relative group overflow-hidden">
      
      {/* Subtle background glow */}
      <div className="absolute top-0 right-0 -mr-16 -mt-16 w-32 h-32 bg-accent opacity-[0.03] blur-3xl pointer-events-none transition-all duration-700 group-hover:opacity-[0.08]" />

      {/* Tooltip */}
      <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-xs px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none whitespace-nowrap shadow-xl z-20">
        Weighted evaluation of your introduction
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></div>
      </div>

      <div className="flex w-full items-center justify-between pointer-events-none mb-6">
        <h3 className="text-[11px] font-black tracking-[0.2em] uppercase text-slate-400">
          OVERALL SCORE
        </h3>
        <span className={`px-4 py-2 rounded-full text-xs font-black tracking-widest uppercase border shadow-sm ${colorClass}`}>
          {label}
        </span>
      </div>

      <div className="flex flex-col flex-1 items-center justify-center w-full mt-2 lg:scale-110">
         <div className="flex items-baseline gap-2">
            <span className="text-8xl md:text-9xl font-black tracking-tighter text-slate-800 dark:text-white tabular-nums leading-none">
              {displayed}
            </span>
            <span className="text-2xl font-semibold text-slate-300">
              / 100
            </span>
         </div>
      </div>

      <div className="w-full h-4 rounded-full bg-slate-100 dark:bg-zinc-900 mt-10 overflow-hidden shadow-inner border border-slate-50">
         <div
           className="h-full rounded-full"
           style={{
             width: `${displayed}%`,
             backgroundImage:
               score >= 80 ? 'linear-gradient(to right, #34d399, #10b981)' :
               score >= 60 ? 'linear-gradient(to right, #fbbf24, #f59e0b)' :
               'linear-gradient(to right, #f87171, #ef4444)'
           }}
         />
      </div>
    </div>
  );
}
