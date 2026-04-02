import { useEffect, useState } from 'react';

interface ConfidenceMeterProps {
  confidence: number; // 0–100
}

/**
 * ConfidenceMeter – Premium circular meter using conic-gradient.
 */
export default function ConfidenceMeter({ confidence }: ConfidenceMeterProps) {
  const [filled, setFilled] = useState(0);

  // Smooth count-up animation 1s
  useEffect(() => {
    let startTimestamp: number | null = null;
    const duration = 1000; // 1s

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // easeOutCubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setFilled(Math.round(eased * confidence));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }, [confidence]);

  const levelColor = 
    confidence >= 70 ? '#2563eb' : 
    confidence >= 40 ? '#f97316' : 
    '#ef4444';

  const label = 
    confidence >= 70 ? 'High' : 
    confidence >= 40 ? 'Medium' : 
    'Low';

  const colorClass = 
    confidence >= 70 ? 'text-accent bg-blue-50 border-blue-100' : 
    confidence >= 40 ? 'text-orange-600 bg-orange-50 border-orange-100' : 
    'text-error bg-red-50 border-red-100';

  return (
    <div className="saas-card p-10 flex flex-col items-center h-full min-h-[260px] relative group overflow-hidden">
      
      {/* Subtle Bottom Right Corner Light */}
      <div className="absolute -bottom-8 -right-8 w-24 h-24 bg-accent opacity-[0.04] blur-2xl pointer-events-none" />

      {/* Tooltip */}
      <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-xs px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none whitespace-nowrap shadow-xl z-20">
        AI confidence in the accuracy of the transcribed intro
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></div>
      </div>

      {/* Header */}
      <div className="flex w-full items-center justify-between pointer-events-none mb-10">
        <h3 className="text-sm font-bold tracking-wide uppercase text-slate-400">
          Confidence
        </h3>
        <span className={`px-3 py-1.5 rounded-full text-xs font-bold tracking-wider border shadow-sm ${colorClass}`}>
          {label}
        </span>
      </div>

      {/* Modern Circular Meter using conic-gradient */}
      <div className="relative flex-1 flex items-center justify-center w-full">
         <div 
           className="relative w-40 h-40 rounded-full flex items-center justify-center transition-all duration-1000 ease-out shadow-sm"
           style={{
             background: `conic-gradient(${levelColor} ${filled * 3.6}deg, #f1f5f9 0deg)`
           }}
         >
            {/* Mask to create ring effect */}
            <div className="absolute inset-[10%] rounded-full bg-white shadow-inner flex flex-col items-center justify-center">
               <span className="text-5xl md:text-6xl font-black tracking-tighter text-slate-800 tabular-nums leading-none">
                 {filled}%
               </span>
            </div>
         </div>
      </div>
    </div>
  );
}
