/**
 * InterviewReport.tsx — post-call interview debrief screen.
 */
import React from 'react';
import { ArrowLeft, CheckCircle, AlertTriangle, TrendingUp } from 'lucide-react';

interface ReportProps {
  data: any | null;
  terminatedReason: string | null;
  onBack: () => void;
}

function Bar({ value, max = 10, color = 'bg-indigo-500' }: { value: number; max?: number; color?: string }) {
  return (
    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden mt-1">
      <div className={`h-full rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${Math.round((value / max) * 100)}%` }} />
    </div>
  );
}

function ScoreChip({ score }: { score: number }) {
  const color = score >= 7 ? 'text-emerald-400 border-emerald-400/30 bg-emerald-400/10'
    : score >= 5 ? 'text-amber-400 border-amber-400/30 bg-amber-400/10'
    : 'text-red-400 border-red-400/30 bg-red-400/10';
  return (
    <span className={`text-xs font-black px-2 py-0.5 rounded-full border ${color}`}>
      {score.toFixed(1)}/10
    </span>
  );
}

function msToTimestamp(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export default function InterviewReport({ data, terminatedReason, onBack }: ReportProps) {
  if (!data) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center text-white">
        <div className="bg-white/5 border border-white/10 rounded-2xl p-8 text-center max-w-md">
          <p className="text-white/60 mb-6">Report unavailable — connection lost.</p>
          <button onClick={onBack} className="flex items-center gap-2 mx-auto text-white/40 hover:text-white text-sm">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const dims = data.dimension_breakdown ?? {};
  const dimEntries: [string, number][] = [
    ['Communication', dims.communication ?? 5],
    ['Content', dims.content ?? 5],
    ['Structure', dims.structure ?? 5],
    ['Confidence', dims.confidence ?? 5],
    ['Integrity', dims.integrity ?? 10],
  ];
  const integrity = data.integrity ?? {};
  const violations: any[] = integrity.violations ?? [];

  return (
    <div className="min-h-screen bg-black text-white overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-10 space-y-8">

        {/* Header */}
        <div>
          <button onClick={onBack} className="flex items-center gap-2 text-white/40 hover:text-white text-sm mb-6">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </button>
          {terminatedReason && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl mb-6 text-sm flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              Interview terminated: {terminatedReason.replace(/_/g, ' ').toLowerCase()}
            </div>
          )}
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
            <div className="flex items-end gap-4 mb-3">
              <span className="text-6xl font-black text-white">{data.overall_score?.toFixed(1)}</span>
              <span className="text-white/40 text-lg mb-2">/10</span>
            </div>
            <p className="text-white/70 leading-relaxed">{data.verdict}</p>
          </div>
        </div>

        {/* Dimension breakdown */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h2 className="text-xs uppercase tracking-widest text-white/40 font-bold mb-4">Dimension Breakdown</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {dimEntries.map(([label, val]) => (
              <div key={label}>
                <div className="flex justify-between text-sm">
                  <span className="text-white/60">{label}</span>
                  <span className="font-black text-white">{val.toFixed(1)}</span>
                </div>
                <Bar value={val} color={val >= 7 ? 'bg-emerald-500' : val >= 5 ? 'bg-amber-500' : 'bg-red-500'} />
              </div>
            ))}
          </div>
        </div>

        {/* Per-question */}
        {data.per_question?.length > 0 && (
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
            <h2 className="text-xs uppercase tracking-widest text-white/40 font-bold mb-4">Per-Question Feedback</h2>
            <div className="space-y-5">
              {data.per_question.map((q: any, i: number) => (
                <div key={i} className="border-b border-white/5 pb-5 last:border-0 last:pb-0">
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <p className="text-white/80 text-sm font-medium leading-snug">{q.question}</p>
                    <ScoreChip score={q.score} />
                  </div>
                  {q.good && <p className="text-emerald-300 text-xs leading-relaxed mb-1">✓ {q.good}</p>}
                  {q.improve && <p className="text-amber-300 text-xs leading-relaxed">→ {q.improve}</p>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Strengths + Improvement plan */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {data.strengths?.length > 0 && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
              <h2 className="text-xs uppercase tracking-widest text-white/40 font-bold mb-4 flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-emerald-400" /> Strengths
              </h2>
              <ul className="space-y-2">
                {data.strengths.map((s: string, i: number) => (
                  <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                    <span className="text-emerald-400 mt-0.5">•</span>{s}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {data.improvement_plan?.length > 0 && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
              <h2 className="text-xs uppercase tracking-widest text-white/40 font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-3 h-3 text-indigo-400" /> Improvement Plan
              </h2>
              <ul className="space-y-2">
                {data.improvement_plan.map((s: string, i: number) => (
                  <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                    <span className="text-indigo-400 mt-0.5">{i + 1}.</span>{s}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Integrity timeline */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h2 className="text-xs uppercase tracking-widest text-white/40 font-bold mb-4 flex items-center gap-2">
            <Shield className="w-3 h-3" /> Integrity — {integrity.score ?? 100}/100
          </h2>
          {violations.length === 0 ? (
            <p className="text-emerald-400 text-sm flex items-center gap-2"><CheckCircle className="w-4 h-4" /> Clean session ✓</p>
          ) : (
            <div className="space-y-2">
              {violations.map((v: any, i: number) => (
                <div key={i} className="flex items-center gap-3 text-xs">
                  <span className="text-white/30 font-mono w-10 shrink-0">{msToTimestamp(v.ts_ms)}</span>
                  <span className={`font-bold uppercase tracking-wide ${v.severity === 'critical' ? 'text-red-400' : 'text-amber-400'}`}>
                    {v.type.replace(/_/g, ' ')}
                  </span>
                  <span className="text-white/30">{v.action}</span>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

// Need Shield for integrity section
function Shield({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
