import { useState } from 'react';

interface FeedbackPanelProps {
  feedback: string;
}

function FeedbackItem({ text, type }: { text: string; type: 'strength' | 'improvement' }) {
  const [expanded, setExpanded] = useState(true); // Default expanded
  const isStrength = type === 'strength';

  return (
    <li 
      onClick={() => setExpanded(!expanded)}
      className="flex flex-col p-3 rounded-xl bg-white/40 hover:bg-white/80 transition-all border border-transparent hover:border-white/50 shadow-sm hover:-translate-y-0.5 hover:shadow-md cursor-pointer group"
    >
      <div className="flex gap-4 items-start">
        <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5 transition-colors ${isStrength ? 'bg-green-100 text-success group-hover:bg-green-200' : 'bg-red-100 text-error group-hover:bg-red-200'}`}>
          {isStrength ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12l7 7 7-7"/></svg>
          )}
        </div>
        <p className={`text-[15px] leading-relaxed text-slate-800 font-semibold flex-1 transition-all ${expanded ? '' : 'line-clamp-2'}`}>
          {text}
        </p>
        <svg 
          className={`w-4 h-4 text-slate-400 transition-transform duration-300 flex-shrink-0 mt-1 ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </li>
  );
}

/**
 * FeedbackPanel – shows up to 8 strengths and 8 improvements,
 * all expanded by default for maximum visibility.
 */
export default function FeedbackPanel({ feedback }: FeedbackPanelProps) {
  // Parse lines: ✔ for positive, ✦ for improvements
  const lines = feedback.split('\n').map(line => line.trim()).filter(Boolean);
  const strengths = lines.filter(l => l.startsWith('✔')).map(l => l.replace('✔', '').trim());
  const improvements = lines.filter(l => l.startsWith('✦')).map(l => l.replace('✦', '').trim());

  return (
    <div className="saas-card p-6 md:p-8 mt-6 animate-in fade-in slide-in-from-bottom-6 duration-500 delay-300 fill-mode-both">
      
      {/* Header */}
      <div className="flex items-center gap-3 mb-8 pb-4 border-b border-border">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
          <path d="M14 2H6a2 2 0 0 0-2 2v16c0 1.1.9 2 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
          <path d="M14 3v5h5M16 13H8M16 17H8M10 9H8"/>
        </svg>
        <span className="font-semibold text-sm uppercase tracking-wider text-muted">
          AI Feedback Report
        </span>
        <span className="ml-auto text-xs font-bold text-slate-400">
          {strengths.length + improvements.length} insights
        </span>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        
        {/* Strengths Column */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2 px-2">
            <span className="w-2.5 h-2.5 rounded-full bg-success opacity-80" />
            <h4 className="text-sm font-bold tracking-wide uppercase text-slate-700">
              Key Strengths
            </h4>
            <span className="ml-auto text-xs font-bold text-green-600 bg-green-50 px-2 py-0.5 rounded-full">
              {strengths.length}
            </span>
          </div>
          <div className="bg-[#ecfdf5] border border-[#bbf7d0] rounded-2xl p-5 grow shadow-sm">
            {strengths.length > 0 ? (
              <ul className="space-y-3">
                {strengths.map((str, i) => (
                  <FeedbackItem key={i} text={str} type="strength" />
                ))}
              </ul>
            ) : (
              <p className="text-sm italic text-slate-500 opacity-75 p-3">
                No distinct strengths detected.
              </p>
            )}
          </div>
        </div>

        {/* Improvements Column */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2 px-2">
            <span className="w-2.5 h-2.5 rounded-full bg-error opacity-80" />
            <h4 className="text-sm font-bold tracking-wide uppercase text-slate-700">
              Focus Areas
            </h4>
            <span className="ml-auto text-xs font-bold text-red-600 bg-red-50 px-2 py-0.5 rounded-full">
              {improvements.length}
            </span>
          </div>
          <div className="bg-[#fef2f2] border border-[#fecaca] rounded-2xl p-5 grow shadow-sm">
            {improvements.length > 0 ? (
              <ul className="space-y-3">
                {improvements.map((imp, i) => (
                  <FeedbackItem key={i} text={imp} type="improvement" />
                ))}
              </ul>
            ) : (
              <p className="text-sm italic text-slate-500 opacity-75 p-3">
                No major area for improvement detected.
              </p>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
