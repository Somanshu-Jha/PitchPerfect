import { useRef } from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CheckCircle2, XCircle, Download, LogOut } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

export default function ResultScreen({ resultData, onRetry, onHistory }: { resultData: any, onRetry: () => void, onHistory: () => void }) {
  const reportRef = useRef<HTMLDivElement>(null);
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    onRetry(); // Navigate back to landing (which shows login when logged out)
  };

  const handleExportPDF = async () => {
    if (!reportRef.current) return;
    try {
      const html2pdf = (await import('html2pdf.js')).default;
      html2pdf()
        .set({
          margin: [10, 10, 10, 10],
          filename: `introlytics_report_${Date.now()}.pdf`,
          image: { type: 'jpeg', quality: 0.98 },
          html2canvas: { scale: 2, backgroundColor: '#0f172a', useCORS: true },
          jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        })
        .from(reportRef.current)
        .save();
    } catch (err) {
      console.error('[PDF Export] Failed:', err);
    }
  };

  // --- Score: backend returns overall_score out of 10, normalize to 100
  const rawScore = resultData?.scores?.overall_score ?? resultData?.score ?? 6;
  const score = Math.round(rawScore <= 10 ? rawScore * 10 : rawScore);

  // --- Confidence: use dynamic_confidence DIRECTLY from backend (0-100)
  // Backend computes this from 5-factor audio analysis (speech rate, pauses, pitch, energy, fluency)
  // No static mapping — always the real computed value
  const dynamicConfidenceNumeric: number | undefined = resultData?.confidence?.dynamic_confidence;
  const confidence: number = (typeof dynamicConfidenceNumeric === 'number' && !isNaN(dynamicConfidenceNumeric))
    ? Math.round(Math.max(0, Math.min(100, dynamicConfidenceNumeric)))
    : 50;  // neutral fallback only if backend failed to provide

  // Derive label from actual numeric value (not from backend label)
  const confidenceLabel = confidence >= 75 ? 'high' : confidence >= 45 ? 'medium' : 'low';

  // --- Feedback: backend returns { positives: [...], improvements: [...] } OR array
  let positives: string[] = [];
  let improvements: string[] = [];

  const feedbackRaw = resultData?.feedback;
  if (feedbackRaw && typeof feedbackRaw === 'object' && !Array.isArray(feedbackRaw)) {
    positives = feedbackRaw.positives ?? [];
    improvements = feedbackRaw.improvements ?? [];
  } else if (Array.isArray(feedbackRaw)) {
    improvements = feedbackRaw;
  }

  // --- Transcript: always show full text (backend sends complete joined text)
  const transcript = resultData?.refined_transcript || resultData?.raw_transcript || '';

  const verdict = score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Needs Work';
  const verdictColor = score >= 80 ? 'text-emerald-300' : score >= 60 ? 'text-indigo-200' : 'text-amber-300';

  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-4xl flex flex-col gap-6 py-8" ref={reportRef}>

      {/* ── Header with logout ─────────────────────────────────────── */}
      <div className="flex justify-between items-end mb-2 px-2">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-slate-100">Interview Analysis</h2>
          <p className="text-slate-400 mt-1">Review your AI-generated performance metrics.</p>
        </div>
        {user && (
          <Button
            id="result-logout-btn"
            variant="outline"
            size="sm"
            onClick={handleLogout}
            className="border-slate-700 text-slate-400 hover:text-red-400 hover:border-red-500/50 hover:bg-red-500/5 transition-all gap-2"
          >
            <LogOut className="w-4 h-4" />
            Log Out
          </Button>
        )}
      </div>

      {/* Score + Confidence */}
      <div className="grid md:grid-cols-2 gap-6">
        
        {/* Score Ring */}
        <Card className="glass-panel p-8 flex flex-col items-center gap-5 border-indigo-500/20 shadow-[0_0_40px_rgba(99,102,241,0.1)] relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent pointer-events-none" />
          <h3 className="text-xs uppercase tracking-[0.2em] text-indigo-300 font-semibold z-10">Overall Score</h3>
          <div className="relative w-40 h-40 flex items-center justify-center z-10">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" className="text-slate-800" />
              <motion.circle
                cx="50" cy="50" r="45"
                fill="none"
                stroke="url(#grad)"
                strokeWidth="8"
                strokeLinecap="round"
                initial={{ strokeDasharray: "0 283" }}
                animate={{ strokeDasharray: `${(score / 100) * 283} 283` }}
                transition={{ duration: 1.5, ease: "easeOut" }}
              />
              <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#a855f7" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="text-4xl font-bold text-white">{score}</span>
              <span className="text-xs text-slate-400 font-medium">/ 100</span>
            </div>
          </div>
          <span className={`text-base font-semibold z-10 ${verdictColor}`}>{verdict}</span>
        </Card>

        {/* Confidence + Transcript */}
        <Card className="glass-panel p-8 flex flex-col gap-6">
          <h3 className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold">Confidence Level</h3>
          <div className="flex flex-col items-center justify-center gap-3">
            <div className="relative w-28 h-28 flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="10" className="text-slate-800" />
                <motion.circle
                  cx="50" cy="50" r="42"
                  fill="none"
                  stroke={confidence >= 75 ? '#22d3ee' : confidence >= 45 ? '#a78bfa' : '#f87171'}
                  strokeWidth="10"
                  strokeLinecap="round"
                  initial={{ strokeDasharray: "0 264" }}
                  animate={{ strokeDasharray: `${(confidence / 100) * 264} 264` }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-2xl font-bold text-white">{confidence}%</span>
              </div>
            </div>
            <span className={`text-sm font-semibold uppercase tracking-widest ${
              confidenceLabel === 'high' ? 'text-cyan-300' :
              confidenceLabel === 'medium' ? 'text-violet-300' : 'text-rose-300'
            }`}>{confidenceLabel}</span>
            <p className="text-slate-500 text-xs text-center">Computed from speech rate, pauses, pitch & energy</p>
          </div>

          {/* Full transcript — scrollable, no truncation */}
          {transcript && (
            <div className="bg-slate-900/50 p-3 rounded-xl border border-slate-800 mt-1">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-2 font-semibold">Your Transcript</p>
              <div className="max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                <p className="text-slate-300 text-sm leading-relaxed italic">"{transcript}"</p>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Feedback */}
      <Card className="glass-panel p-8 space-y-6 relative overflow-hidden">
        <div className="absolute right-0 top-0 w-64 h-64 bg-purple-500/5 blur-[80px] rounded-full pointer-events-none" />
        <h3 className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold z-10 relative">AI Feedback</h3>
        
        <div className="grid md:grid-cols-2 gap-6 z-10 relative">
          {/* Positives */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" /> What You Did Well
            </h4>
            {positives.length > 0 ? positives.map((item, i) => (
              <div key={i} className="flex gap-3 items-start bg-emerald-500/5 border border-emerald-500/20 p-3 rounded-xl">
                <div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                <p className="text-slate-300 text-sm leading-relaxed">{item}</p>
              </div>
            )) : (
              <p className="text-slate-500 text-sm italic">No positive feedback recorded.</p>
            )}
          </div>

          {/* Improvements */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center gap-2">
              <XCircle className="w-4 h-4" /> Areas to Improve
            </h4>
            {improvements.length > 0 ? improvements.map((item, i) => (
              <div key={i} className="flex gap-3 items-start bg-rose-500/5 border border-rose-500/20 p-3 rounded-xl">
                <div className="w-1.5 h-1.5 mt-1.5 rounded-full bg-rose-400 flex-shrink-0" />
                <p className="text-slate-300 text-sm leading-relaxed">{item}</p>
              </div>
            )) : (
              <p className="text-slate-500 text-sm italic">All areas covered!</p>
            )}
          </div>
        </div>
      </Card>

      {/* Coaching Summary (from GenAI) */}
      {resultData?.feedback?.coaching_summary && (
        <Card className="glass-panel p-6 relative overflow-hidden">
          <div className="absolute left-0 top-0 w-40 h-40 bg-indigo-500/5 blur-[60px] rounded-full pointer-events-none" />
          <h3 className="text-xs uppercase tracking-[0.2em] text-indigo-300 font-semibold mb-3 z-10 relative">AI Coaching Summary</h3>
          <p className="text-slate-300 text-sm leading-relaxed z-10 relative">{resultData.feedback.coaching_summary}</p>
        </Card>
      )}

      <div className="flex justify-end gap-4 mt-2">
        <Button variant="outline" size="lg" className="border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white" onClick={handleExportPDF}>
          <Download className="mr-2 h-4 w-4" /> Export PDF
        </Button>
        <Button variant="outline" size="lg" className="border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white" onClick={onHistory}>View History</Button>
        <Button size="lg" className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white border-0 shadow-lg shadow-indigo-500/20" onClick={onRetry}>Try Again</Button>
      </div>
    </motion.div>
  );
}
