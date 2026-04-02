import ScoreCard from '../ScoreCard';
import ConfidenceMeter from '../ConfidenceMeter';
import FeedbackPanel from '../FeedbackPanel';
import FadeIn from '../FadeIn';

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
  } | null;
  isLoading: boolean;
  onRetry: () => void;
}

/**
 * ResultSection – full performance dashboard with transcript, scores,
 * audio analysis breakdown, and detailed feedback.
 */
export default function ResultSection({ data, isLoading, onRetry }: ResultSectionProps) {
  
  if (isLoading) {
    return (
      <section className="min-h-screen py-24 flex items-center justify-center bg-transparent w-full">
        <div className="flex flex-col items-center gap-6">
          <div className="relative w-16 h-16 flex items-center justify-center">
             <div className="absolute inset-0 rounded-full border-4 border-slate-100" />
             <div className="absolute inset-0 rounded-full border-4 border-accent border-t-transparent animate-spin" />
          </div>
          <span className="text-xl font-black tracking-widest text-accent uppercase animate-pulse">
             Extracting Insights...
          </span>
        </div>
      </section>
    );
  }

  if (!data) {
    return (
      <section id="results" className="min-h-[60vh] py-24 px-6 md:px-12 w-full flex items-center justify-center">
        <div className="w-full max-w-6xl mx-auto border-2 border-dashed border-slate-200 rounded-[32px] p-12 md:p-20 flex flex-col items-center text-center gap-8 bg-white/40">
           <div className="w-20 h-20 rounded-full bg-slate-50 flex items-center justify-center text-slate-300">
             <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
               <path d="M12 20h.01"/><path d="M12 16h.01"/><path d="M12 12h.01"/><path d="M12 8h.01"/><path d="M12 4h.01"/>
             </svg>
           </div>
           <div className="space-y-3">
             <h2 className="text-3xl font-black text-slate-400 tracking-tight">Performance Report</h2>
             <p className="text-slate-400 font-bold uppercase tracking-widest text-xs">Record your introduction to see analysis</p>
           </div>
           <div className="grid grid-cols-2 gap-8 w-full max-w-md opacity-40 grayscale">
              <div className="p-6 rounded-2xl bg-white border border-slate-100 flex flex-col gap-1 items-center">
                <span className="text-xs font-black text-slate-400 uppercase">Score</span>
                <span className="text-4xl font-black text-slate-300">--</span>
              </div>
              <div className="p-6 rounded-2xl bg-white border border-slate-100 flex flex-col gap-1 items-center">
                <span className="text-xs font-black text-slate-400 uppercase">Confidence</span>
                <span className="text-4xl font-black text-slate-300">--</span>
              </div>
           </div>
        </div>
      </section>
    );
  }

  const dlMetrics = data.dlMetrics || {};
  const audioFeatures = data.audioFeatures || {};
  const audioReasoning = data.audioReasoning || {};

  return (
    <section id="results" className="min-h-screen py-24 px-6 md:px-12 w-full">
      <div className="w-full max-w-6xl mx-auto flex flex-col gap-12">

        <FadeIn delay={0} yOffset={20}>
          <div className="text-center md:text-left space-y-4 mb-4">
            <h2 className="text-5xl md:text-6xl font-black tracking-tighter text-slate-800">
              Your Performance
            </h2>
            <p className="text-xl text-slate-500 font-medium max-w-2xl">
              Deep analysis of your interview introduction — transcript, vocal analysis, and detailed scoring.
            </p>
          </div>
        </FadeIn>

        {/* Transcript Card */}
        {data.transcript && (
          <FadeIn delay={50} yOffset={20}>
            <div className="saas-card p-6 md:p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center gap-3 mb-5 pb-3 border-b border-border">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" />
                </svg>
                <span className="font-semibold text-sm uppercase tracking-wider text-muted">
                  Your Transcript
                </span>
                <span className="ml-auto text-xs font-bold text-slate-400">
                  {data.transcript.split(/\s+/).filter(Boolean).length} words
                </span>
              </div>
              <div className="text-[16px] leading-relaxed text-slate-700 font-medium whitespace-pre-wrap break-words">
                {data.transcript}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Score + Confidence Row */}
        <div className="grid md:grid-cols-2 gap-10">
          <FadeIn delay={150} yOffset={30}>
            <ScoreCard score={data.score} />
          </FadeIn>
          <FadeIn delay={300} yOffset={30}>
            <ConfidenceMeter confidence={data.confidence} />
          </FadeIn>
        </div>

        {/* DL Metrics Breakdown (if available) */}
        {Object.keys(dlMetrics).length > 0 && (
          <FadeIn delay={350} yOffset={20}>
            <div className="saas-card p-6 md:p-8">
              <div className="flex items-center gap-3 mb-6 pb-3 border-b border-border">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-500">
                  <path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>
                </svg>
                <span className="font-semibold text-sm uppercase tracking-wider text-muted">
                  Score Breakdown
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { key: 'clarity', label: 'Clarity', color: 'bg-blue-500' },
                  { key: 'confidence', label: 'Confidence', color: 'bg-green-500' },
                  { key: 'structure', label: 'Structure', color: 'bg-purple-500' },
                  { key: 'tone', label: 'Tone', color: 'bg-orange-500' },
                  { key: 'fluency', label: 'Fluency', color: 'bg-teal-500' },
                  { key: 'dl_overall', label: 'Overall', color: 'bg-slate-700' },
                ].map(({ key, label, color }) => {
                  const val = dlMetrics[key];
                  if (val === undefined) return null;
                  const pct = Math.round((val / 10) * 100);
                  return (
                    <div key={key} className="bg-white/60 rounded-xl p-4 border border-slate-100 shadow-sm">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</span>
                        <span className="text-lg font-black text-slate-800">{val}</span>
                      </div>
                      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${color} rounded-full transition-all duration-1000 ease-out`} 
                          style={{ width: `${pct}%` }} 
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Audio Analysis Reasoning */}
        {Object.keys(audioReasoning).length > 0 && (
          <FadeIn delay={400} yOffset={20}>
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
                  <div key={key} className="flex gap-3 p-4 rounded-xl bg-white/50 border border-slate-100">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-teal-50 flex items-center justify-center">
                      <span className="text-teal-600 text-xs font-black uppercase">{key.charAt(0)}</span>
                    </div>
                    <div>
                      <h5 className="text-xs font-black uppercase tracking-wide text-slate-500 mb-1">{key}</h5>
                      <p className="text-sm text-slate-700 leading-relaxed">{reasoning}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Coaching Summary */}
        {data.coachingSummary && (
          <FadeIn delay={420} yOffset={20}>
            <div className="p-5 bg-gradient-to-r from-accent/5 to-purple-500/5 rounded-2xl border border-accent/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-accent text-lg">💡</span>
                <span className="text-xs font-black uppercase tracking-wide text-accent">Coaching Summary</span>
              </div>
              <p className="text-sm text-slate-700 font-medium leading-relaxed">{data.coachingSummary}</p>
            </div>
          </FadeIn>
        )}

        {/* Feedback Panel */}
        <FadeIn delay={450} yOffset={30}>
          <FeedbackPanel feedback={data.feedback} />
        </FadeIn>

        {/* Retry Button */}
        <FadeIn delay={600} yOffset={20} className="mt-8 flex justify-end">
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
