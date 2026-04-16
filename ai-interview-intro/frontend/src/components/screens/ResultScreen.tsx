import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CheckCircle2, XCircle } from 'lucide-react';

export default function ResultScreen({ resultData, onRetry, onHistory }: { resultData: any, onRetry: () => void, onHistory: () => void }) {
  // --- Score: backend returns overall_score out of 10, normalize to 100
  const rawScore = resultData?.scores?.overall_score ?? resultData?.score ?? 6;
  const score = Math.round(rawScore <= 10 ? rawScore * 10 : rawScore);

  const [showFeedback, setShowFeedback] = useState(false);
  const [overrideScore, setOverrideScore] = useState(10);
  const [feedbackReason, setFeedbackReason] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<"idle"|"loading"|"success"|"error">("idle");
  const [feedbackMsg, setFeedbackMsg] = useState("");

  const submitFeedback = async () => {
    if (feedbackReason.split(" ").length < 3) {
      setFeedbackStatus("error");
      setFeedbackMsg("Please provide a detailed reason (at least 3 words) so the AI can learn.");
      return;
    }
    setFeedbackStatus("loading");
    try {
      const response = await fetch("http://localhost:8000/student/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Use a generic user_id for beta testing locally if not provided
        body: JSON.stringify({
          user_id: "beta_tester",
          transcript: resultData?.refined_transcript || resultData?.raw_transcript || "",
          features: resultData?.features || [],
          original_scores: resultData?.scores || {},
          new_scores: { ...resultData?.scores, overall: overrideScore },
          reason: feedbackReason
        })
      });
      const data = await response.json();
      if (data.status === "accepted") {
        setFeedbackStatus("success");
      } else {
        setFeedbackStatus("error");
        setFeedbackMsg(data.message || "Feedback rejected by AI Gatekeeper.");
      }
    } catch(err) {
      setFeedbackStatus("error");
      setFeedbackMsg("Failed to connect to server.");
    }
  };


  // --- Confidence: string like "medium", "high", "low" → map to percentage
  const confidenceRaw: string = resultData?.scores?.confidence ?? 'medium';
  const confidenceMap: Record<string, number> = { low: 30, medium: 60, high: 90 };
  const confidence = confidenceMap[confidenceRaw.toLowerCase()] ?? 60;

  // --- Feedback: backend returns { positives: [{text, evidence}, ...], improvements: [{text, evidence}, ...] }  OR array
  let positives: string[] = [];
  let improvements: string[] = [];

  const extractText = (items: any[]): string[] => {
    return items.map((item: any) => typeof item === 'string' ? item : item?.text ?? String(item));
  };

  const feedbackRaw = resultData?.feedback;
  if (feedbackRaw && typeof feedbackRaw === 'object' && !Array.isArray(feedbackRaw)) {
    positives = extractText(feedbackRaw.positives ?? []);
    improvements = extractText(feedbackRaw.improvements ?? []);
  } else if (Array.isArray(feedbackRaw)) {
    // flat array fallback
    improvements = extractText(feedbackRaw);
  }

  // --- Transcript
  const transcript = resultData?.refined_transcript || resultData?.raw_transcript || '';

  const verdict = score >= 75 ? 'Excellent' : score >= 50 ? 'Good' : 'Needs Work';

  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-4xl flex flex-col gap-6 py-8">

      <div className="flex justify-between items-end mb-2 px-2">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-slate-100">Interview Analysis</h2>
          <p className="text-slate-400 mt-1">Review your AI-generated performance metrics.</p>
        </div>
      </div>

      {/* Score + Confidence */}
      <div className="grid md:grid-cols-2 gap-6">
        
        {/* Score Ring */}
        <Card className="glass-panel p-8 flex flex-col items-center gap-5 border-indigo-500/20 shadow-[0_0_40px_rgba(99,102,241,0.1)] relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent pointer-events-none" />
          <h3 className="text-xs uppercase tracking-[0.2em] text-indigo-300 font-semibold z-10">Overall Score</h3>
          <div className="relative w-40 h-40 flex items-center justify-center z-10">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" className="text-slate-800 dark:text-white" />
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
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-5xl font-bold text-white leading-none">{score}</span>
              <span className="text-sm text-slate-400 font-medium">/ 100</span>
            </div>
          </div>
          <span className="text-base font-semibold text-indigo-200 z-10">{verdict}</span>
        </Card>

        {/* Confidence + Transcript */}
        <Card className="glass-panel p-8 flex flex-col gap-6">
          <h3 className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold">Confidence Level</h3>
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <div className="relative w-28 h-28 flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="10" className="text-slate-800 dark:text-white" />
                <motion.circle
                  cx="50" cy="50" r="42"
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth="10"
                  strokeLinecap="round"
                  initial={{ strokeDasharray: "0 264" }}
                  animate={{ strokeDasharray: `${(confidence / 100) * 264} 264` }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-4xl font-bold text-white">{confidence}%</span>
              </div>
            </div>
            <span className="text-sm font-semibold uppercase tracking-widest text-cyan-300">{confidenceRaw}</span>
          </div>
          {transcript && (
            <div className="bg-slate-900/50 p-3 rounded-xl border border-slate-800 mt-2">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-2 font-semibold">Your Transcript</p>
              <p className="text-slate-300 text-sm leading-relaxed italic line-clamp-4">"{transcript}"</p>
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
              <p className="text-slate-500 dark:text-gray-400 text-sm italic">No positive feedback recorded.</p>
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
              <p className="text-slate-500 dark:text-gray-400 text-sm italic">All areas covered!</p>
            )}
          </div>
        </div>
      </Card>

      <div className="flex justify-between items-center mt-2">
        <Button variant="ghost" className="text-amber-400 hover:text-amber-300 hover:bg-amber-400/10" onClick={() => setShowFeedback(true)}>
          ⚠️ Rate this Score (Beta Testers)
        </Button>
        <div className="flex justify-end gap-4">
          <Button variant="outline" size="lg" className="border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white" onClick={onHistory}>View History</Button>
          <Button size="lg" className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white border-0 shadow-lg shadow-indigo-500/20" onClick={onRetry}>Try Again</Button>
        </div>
      </div>

      <AnimatePresence>
        {showFeedback && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} 
            className="mt-6 bg-slate-900 border border-amber-500/30 p-6 rounded-xl relative">
            <button className="absolute top-4 right-4 text-slate-400 hover:text-white" onClick={() => setShowFeedback(false)}>✕</button>
            <h3 className="text-lg font-bold text-amber-400 mb-2">Help Train the AI</h3>
            <p className="text-sm text-slate-300 mb-4">If you feel the AI evaluated this incorrectly, provide the correct score and explain why. The DeepSeek Gatekeeper will validate your reasoning before injecting it into the training pipeline.</p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold mb-1 text-slate-200">What should the True Score be? (1-10)</label>
                <input type="number" min="1" max="10" step="0.5" className="bg-slate-800 border border-slate-700 rounded-md p-2 w-32 text-white" 
                  value={overrideScore} onChange={e => setOverrideScore(Number(e.target.value))} />
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1 text-slate-200">Why was the AI wrong? (Required)</label>
                <textarea rows={3} className="w-full bg-slate-800 border border-slate-700 rounded-md p-3 text-white placeholder:text-slate-500" 
                  placeholder="E.g. The audio was stuttering heavily but the AI gave me a high score anyway..."
                  value={feedbackReason} onChange={e => setFeedbackReason(e.target.value)}></textarea>
              </div>
              
              <div className="flex items-center gap-4">
                <Button onClick={submitFeedback} disabled={feedbackStatus === "loading"} className="bg-amber-600 hover:bg-amber-500">
                  {feedbackStatus === "loading" ? "Analyzing..." : "Submit Feedback"}
                </Button>
                {feedbackStatus === "success" && <span className="text-emerald-400 flex items-center gap-1"><CheckCircle2 className="w-4 h-4" /> Added to Golden Dataset!</span>}
                {feedbackStatus === "error" && <span className="text-rose-400 font-semibold">{feedbackMsg}</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </motion.div>

  );
}
