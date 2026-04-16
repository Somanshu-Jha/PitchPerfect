import { useState, useEffect } from 'react';
import ScoreCard from '../ScoreCard';
import ConfidenceMeter from '../ConfidenceMeter';
import FadeIn from '../FadeIn';
import InsightGame from '../InsightGame';

interface ResultSectionProps {
  data: {
    score: number;
    confidence: number;
    feedback: string;
    transcript?: string;
    audioReasoning?: Record<string, string>;
    dlMetrics?: Record<string, number>;
    audioFeatures?: Record<string, any>;
    coachingSummary?: string;
    rubricBreakdown?: any[];
    positives?: string[];
    improvements?: string[];
    suggestions?: string[];
    scoreDeductionReason?: string;
    rubricScore?: number | null;
    dlRawScore?: number | null;
    resumeMatched?: string[];
    resumeMissed?: string[];
  } | null;
  isLoading: boolean;
  onRetry: () => void;
  resumeHint?: string;
  transcriptHint?: string;
}

/** Category badge colors */
const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'RESUME GAP': { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  'DELIVERY': { bg: 'bg-sky-50', text: 'text-sky-700', border: 'border-sky-200' },
  'CONTENT DEPTH': { bg: 'bg-violet-50', text: 'text-violet-700', border: 'border-violet-200' },
  'STRUCTURE': { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
  'PROFESSIONAL POLISH': { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
};

/** Extract [CATEGORY] tag from text */
function parseCategory(text: string): { category: string | null; cleanText: string } {
  const match = text.match(/^\s*\[([A-Z\s]+)\]\s*/);
  if (match) {
    return { category: match[1].trim(), cleanText: text.replace(match[0], '').trim() };
  }
  return { category: null, cleanText: text };
}

/** Expandable feedback item with optional category badge */
function FeedbackItem({ text, type }: { text: string; type: 'strength' | 'improvement' }) {
  const [expanded, setExpanded] = useState(true);
  const isStrength = type === 'strength';
  const { category, cleanText } = parseCategory(text);
  const catColors = category ? CATEGORY_COLORS[category] || { bg: 'bg-gray-50', text: 'text-gray-600 dark:text-gray-300', border: 'border-gray-200' } : null;

  return (
    <li 
      onClick={() => setExpanded(!expanded)}
      className="flex flex-col p-3 rounded-xl bg-white/40 hover:bg-white/80 transition-all border border-transparent hover:border-white/50 shadow-sm hover:-translate-y-0.5 hover:shadow-md cursor-pointer group"
    >
      <div className="flex gap-4 items-start">
        <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5 transition-colors ${isStrength ? 'bg-green-100 text-green-600 group-hover:bg-green-200' : 'bg-red-100 text-red-500 group-hover:bg-red-200'}`}>
          {isStrength ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12l7 7 7-7"/></svg>
          )}
        </div>
        <div className="flex-1 flex flex-col gap-1.5">
          {category && catColors && (
            <span className={`self-start inline-flex px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-wider ${catColors.bg} ${catColors.text} border ${catColors.border}`}>
              {category}
            </span>
          )}
          <p className={`text-[15px] leading-relaxed text-slate-800 dark:text-white font-semibold transition-all ${expanded ? '' : 'line-clamp-2'}`}>
            {cleanText}
          </p>
        </div>
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
 * ResultSection – performance dashboard matching the original two-column 
 * "AI Feedback Report" layout with Key Strengths and Focus Areas side-by-side.
 * Now includes an interactive Chrome Dino-style game during the loading state.
 */
export default function ResultSection({ data, isLoading, onRetry, resumeHint, transcriptHint }: ResultSectionProps) {
  const [gameWaiting, setGameWaiting] = useState(false);
  
  // When analysis starts, activate the game waiting mode
  // This keeps the game visible even after analysis completes
  useEffect(() => {
    if (isLoading) {
      setGameWaiting(true);
    }
  }, [isLoading]);
  
  // When analysis completes but user is still in the game, hold results
  const analysisReady = !isLoading && data !== null && gameWaiting;
  
  if (isLoading || gameWaiting) {
    return (
      <InsightGame
        resumeHint={resumeHint}
        transcriptHint={transcriptHint}
        analysisComplete={!isLoading && data !== null}
        onViewResults={() => setGameWaiting(false)}
      />
    );
  }

  if (!data) {
    return (
      <section id="results" className="min-h-[60vh] py-24 px-6 md:px-12 w-full flex items-center justify-center">
        <div className="w-full max-w-6xl mx-auto border-2 border-dashed border-slate-200 dark:border-white/20 rounded-[32px] p-12 md:p-20 flex flex-col items-center text-center gap-8 bg-white/40">
           <div className="w-20 h-20 rounded-full bg-slate-50 dark:bg-black flex items-center justify-center text-slate-300">
             <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
               <path d="M12 20h.01"/><path d="M12 16h.01"/><path d="M12 12h.01"/><path d="M12 8h.01"/><path d="M12 4h.01"/>
             </svg>
           </div>
           <div className="space-y-3">
             <h2 className="text-3xl font-black text-slate-400 tracking-tight">Performance Report</h2>
             <p className="text-slate-400 font-bold uppercase tracking-widest text-xs">Record your introduction to see analysis</p>
           </div>
           <div className="grid grid-cols-2 gap-8 w-full max-w-md opacity-40 grayscale">
              <div className="p-6 rounded-2xl bg-white dark:bg-black border border-slate-100 dark:border-white/10 flex flex-col gap-1 items-center">
                <span className="text-xs font-black text-slate-400 uppercase">Score</span>
                <span className="text-4xl font-black text-slate-300">--</span>
              </div>
              <div className="p-6 rounded-2xl bg-white dark:bg-black border border-slate-100 dark:border-white/10 flex flex-col gap-1 items-center">
                <span className="text-xs font-black text-slate-400 uppercase">Confidence</span>
                <span className="text-4xl font-black text-slate-300">--</span>
              </div>
           </div>
        </div>
      </section>
    );
  }

  const audioReasoning = data.audioReasoning || {};
  const positives = data.positives || [];
  const improvements = data.improvements || [];
  const suggestions = data.suggestions || [];
  const resumeMatched = data.resumeMatched || [];
  const resumeMissed = data.resumeMissed || [];
  const totalInsights = positives.length + improvements.length;

  // Clean coaching summary (strip raw JSON/markdown artifacts)
  const cleanCoachingSummary = (raw: string): string => {
    if (!raw) return '';
    let text = raw.trim();
    text = text.replace(/```json\s*/g, '').replace(/```\s*/g, '');
    if (text.startsWith('{')) {
      try {
        const parsed = JSON.parse(text);
        const source = parsed?.coach || parsed?.feedback || parsed;
        return source?.coaching_summary || source?.summary || text;
      } catch {
        const match = text.match(/"coaching_summary"\s*:\s*"([^"]+)"/);
        if (match) return match[1];
      }
    }
    return text;
  };
  const coachingSummary = cleanCoachingSummary(data.coachingSummary || '');
  const hasResumeData = resumeMatched.length > 0 || resumeMissed.length > 0;

  return (
    <section id="results" className="min-h-screen py-24 px-6 md:px-12 w-full">
      <div className="w-full max-w-6xl mx-auto flex flex-col gap-12">

        <FadeIn delay={0} yOffset={20}>
          <div className="text-center md:text-left space-y-4 mb-4">
            <h2 className="text-5xl md:text-6xl font-black tracking-tighter text-slate-800 dark:text-white">
              Your Performance
            </h2>
            <p className="text-xl text-slate-500 dark:text-gray-400 font-medium max-w-2xl">
              Deep analysis of your interview introduction — transcript, vocal analysis, and detailed scoring.
            </p>
          </div>
        </FadeIn>

        {/* Score + Confidence Row */}
        <div className="grid md:grid-cols-2 gap-10">
          <FadeIn delay={150} yOffset={30}>
            <ScoreCard score={data.score} />
          </FadeIn>
          <FadeIn delay={300} yOffset={30}>
            <ConfidenceMeter confidence={data.confidence} />
          </FadeIn>
        </div>

        {/* ═══ COACHING SUMMARY ═══ */}
        {coachingSummary && (
          <FadeIn delay={350} yOffset={20}>
            <div className="saas-card p-6 md:p-8 border-blue-100 bg-gradient-to-br from-blue-50/40 to-indigo-50/30">
              <div className="flex items-center gap-3 mb-5 pb-3 border-b border-blue-100">
                <span className="text-lg">🎯</span>
                <span className="font-semibold text-sm uppercase tracking-wider text-blue-600">AI Coaching Summary</span>
              </div>
              <p className="text-[15px] text-slate-700 dark:text-gray-200 leading-relaxed font-medium">
                {coachingSummary}
              </p>
            </div>
          </FadeIn>
        )}

        {/* ═══ AI FEEDBACK REPORT — Two‑Column Layout (Original Design) ═══ */}
        {(positives.length > 0 || improvements.length > 0) && (
          <FadeIn delay={400} yOffset={20}>
            <div className="saas-card p-6 md:p-8 animate-in fade-in slide-in-from-bottom-6 duration-500">
              
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
                  {totalInsights} insights
                </span>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                
                {/* Strengths Column */}
                <div className="flex flex-col gap-4">
                  <div className="flex items-center gap-2 mb-2 px-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-green-500 opacity-80" />
                    <h4 className="text-sm font-bold tracking-wide uppercase text-slate-700 dark:text-gray-200">
                      Key Strengths
                    </h4>
                    <span className="ml-auto text-xs font-bold text-green-600 bg-green-50 px-2 py-0.5 rounded-full">
                      {positives.length}
                    </span>
                  </div>
                  <div className="bg-[#ecfdf5] border border-[#bbf7d0] rounded-2xl p-5 grow shadow-sm">
                    {positives.length > 0 ? (
                      <ul className="space-y-3">
                        {positives.map((str, i) => (
                          <FeedbackItem key={i} text={str} type="strength" />
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm italic text-slate-500 dark:text-gray-400 opacity-75 p-3">
                        No distinct strengths detected.
                      </p>
                    )}
                  </div>
                </div>

                {/* Focus Areas Column */}
                <div className="flex flex-col gap-4">
                  <div className="flex items-center gap-2 mb-2 px-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 opacity-80" />
                    <h4 className="text-sm font-bold tracking-wide uppercase text-slate-700 dark:text-gray-200">
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
                          <FeedbackItem key={`imp-${i}`} text={imp} type="improvement" />
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm italic text-slate-500 dark:text-gray-400 opacity-75 p-3">
                        No major area for improvement detected.
                      </p>
                    )}
                  </div>
                </div>

              </div>
            </div>
          </FadeIn>
        )}

        {/* ═══ RESUME-PITCH ALIGNMENT ═══ */}
        {hasResumeData && (
          <FadeIn delay={460} yOffset={20}>
            <div className="saas-card p-6 md:p-8 border-indigo-100 bg-indigo-50/20">
              <div className="flex items-center gap-3 mb-5 pb-3 border-b border-indigo-100">
                <span className="text-lg">📄</span>
                <span className="font-semibold text-sm uppercase tracking-wider text-indigo-600">Resume vs Pitch Alignment</span>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                {/* Matched */}
                {resumeMatched.length > 0 && (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center gap-2 px-1">
                      <span className="w-2 h-2 rounded-full bg-emerald-500" />
                      <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">
                        Matched ({resumeMatched.length})
                      </span>
                    </div>
                    <div className="bg-white/60 rounded-xl border border-emerald-100 p-4">
                      <div className="flex flex-wrap gap-2">
                        {resumeMatched.map((item, i) => (
                          <span key={i} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Missed */}
                {resumeMissed.length > 0 && (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center gap-2 px-1">
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      <span className="text-xs font-bold uppercase tracking-wider text-red-500">
                        Missed from Resume ({resumeMissed.length})
                      </span>
                    </div>
                    <div className="bg-white/60 rounded-xl border border-red-100 p-4">
                      <div className="flex flex-wrap gap-2">
                        {resumeMissed.map((item, i) => (
                          <span key={i} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-red-50 text-red-600 border border-red-200">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </FadeIn>
        )}

        {/* ═══ SUGGESTED VARIATIONS ═══ */}
        {suggestions.length > 0 && (
          <FadeIn delay={480} yOffset={20}>
            <div className="saas-card p-6 md:p-8 border-violet-100 bg-violet-50/30">
              <div className="flex items-center gap-3 mb-5 pb-3 border-b border-violet-100">
                <span className="text-lg">✨</span>
                <span className="font-semibold text-sm uppercase tracking-wider text-violet-600">Sentence Variations & Rewrites</span>
              </div>
              <ul className="space-y-4">
                {suggestions.map((sug, i) => (
                  <li key={i} className="flex gap-4 p-4 bg-white/60 border border-violet-100 rounded-xl shadow-sm">
                    <span className="text-violet-500 font-bold mt-0.5">💡</span>
                    <p className="text-sm text-slate-700 dark:text-gray-200 font-medium leading-relaxed italic border-l-2 border-violet-300 pl-3">"{sug}"</p>
                  </li>
                ))}
              </ul>
            </div>
          </FadeIn>
        )}

        {/* ═══ VOICE ANALYSIS ═══ */}
        {Object.keys(audioReasoning).length > 0 && (
          <FadeIn delay={500} yOffset={20}>
            <div className="saas-card p-6 md:p-8">
              <div className="flex items-center gap-3 mb-6 pb-3 border-b border-border">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-teal-500">
                  <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
                </svg>
                <span className="font-semibold text-sm uppercase tracking-wider text-muted">
                  Voice Analysis
                </span>
              </div>
              <div className="grid gap-3">
                {Object.entries(audioReasoning).map(([key, reasoning]) => (
                  <div key={key} className="flex gap-3 p-4 rounded-xl bg-white/50 border border-slate-100 dark:border-white/10">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-teal-50 flex items-center justify-center">
                      <span className="text-teal-600 text-xs font-black uppercase">{key.charAt(0)}</span>
                    </div>
                    <div>
                      <h5 className="text-xs font-black uppercase tracking-wide text-slate-500 dark:text-gray-400 mb-1">{key}</h5>
                      <p className="text-sm text-slate-700 dark:text-gray-200 leading-relaxed">{reasoning}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Retry Button */}
        <FadeIn delay={600} yOffset={20} className="mt-8 flex justify-center">
           <button
             onClick={onRetry}
             className="btn-primary px-10 py-4 shadow-[0_10px_20px_rgba(37,99,235,0.2)] text-base font-bold tracking-tight hover:-translate-y-1 transition-all"
           >
              Try Another Attempt
           </button>
        </FadeIn>

      </div>
    </section>
  );
}
