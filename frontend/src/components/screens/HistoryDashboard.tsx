import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Clock, Activity, TrendingUp, CheckCircle2, AlertCircle, Loader2, Trash2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { useAuth } from '../../contexts/AuthContext';

const API_BASE = 'http://127.0.0.1:5000';

interface ProgressData {
  user_id: string;
  has_data: boolean;
  total_attempts: number;
  improvements_made: number;
  score_delta: number;
  history: { timestamp: string; score: number; fillers: number; confidence?: number }[];
}

interface AttemptRecord {
  attempt_id: number;
  transcript: string;
  score: number;
  feedback: { positives?: string[]; improvements?: string[]; coaching_summary?: string };
  fillers: number;
  timestamp: string;
  improved: boolean;
}

export default function HistoryDashboard({ onBack }: { onBack: () => void }) {
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [attempts, setAttempts] = useState<AttemptRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedAttempt, setExpandedAttempt] = useState<number | null>(null);
  const { user } = useAuth();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const userId = user ? user.user_id : 'local_demo';
        const [progressRes, historyRes] = await Promise.all([
          fetch(`${API_BASE}/student/progress/${userId}`),
          fetch(`${API_BASE}/student/history/${userId}`),
        ]);
        const progressJson = await progressRes.json();
        const historyJson = await historyRes.json();

        if (progressJson?.data?.has_data) setProgress(progressJson.data);
        if (historyJson?.data) setAttempts(historyJson.data);
      } catch (err) {
        console.error('[HistoryDashboard] API fetch failed:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Build chart data from progress history
  const chartData = (progress?.history ?? []).map((h, i) => ({
    attempt: `#${i + 1}`,
    score: Math.round(h.score * 10),
    confidence: Math.round(h.confidence ?? 0),
    fillers: h.fillers,
  }));

  const handleDeleteAttempt = async (attemptId: number) => {
    if (!confirm('Are you sure you want to delete this attempt?')) return;
    try {
      const userId = user ? user.user_id : 'local_demo';
      const res = await fetch(`${API_BASE}/student/history/${userId}/${attemptId}`, { method: 'DELETE' });
      if (res.ok) {
        setAttempts(prev => prev.filter(a => a.attempt_id !== attemptId));
        // Refresh progress by causing re-render (you could re-fetch here if needed)
      }
    } catch (err) {
      console.error('Failed to delete attempt:', err);
    }
  };

  const avgScore = progress
    ? Math.round((progress.history.reduce((sum, h) => sum + h.score, 0) / progress.history.length) * 10)
    : 0;

  if (loading) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center gap-4 py-32">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
        <p className="text-slate-400 text-sm">Loading your performance data...</p>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-4xl flex flex-col gap-8 py-8">
      
      {/* Header */}
      <div className="flex items-center gap-4 border-b border-slate-800 pb-6">
        <Button variant="ghost" size="icon" onClick={onBack} className="rounded-full hover:bg-slate-800 text-slate-400">
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">Performance Analytics</h1>
          <p className="text-slate-400 text-sm mt-1">Track your interview improvement over time.</p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="glass-panel p-5 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-slate-400">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold">Avg Score</span>
          </div>
          <span className="text-3xl font-bold text-white">{avgScore}<span className="text-base text-slate-500">/100</span></span>
        </Card>
        <Card className="glass-panel p-5 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-slate-400">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold">Attempts</span>
          </div>
          <span className="text-3xl font-bold text-white">{progress?.total_attempts ?? 0}</span>
        </Card>
        <Card className="glass-panel p-5 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-slate-400">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold">Improvements</span>
          </div>
          <span className="text-3xl font-bold text-emerald-400">{progress?.improvements_made ?? 0}</span>
        </Card>
        <Card className="glass-panel p-5 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-slate-400">
            <TrendingUp className="w-4 h-4 text-purple-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold">Net Growth</span>
          </div>
          <span className={`text-3xl font-bold ${(progress?.score_delta ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {(progress?.score_delta ?? 0) >= 0 ? '+' : ''}{Math.round((progress?.score_delta ?? 0) * 10)}
          </span>
        </Card>
      </div>

      {/* Performance Chart */}
      {chartData.length > 1 && (
        <Card className="glass-panel p-6 relative overflow-hidden">
          <div className="absolute right-0 top-0 w-48 h-48 bg-indigo-500/5 blur-[80px] rounded-full pointer-events-none" />
          <h3 className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold mb-6 z-10 relative">Score Trend</h3>
          <div className="h-64 w-full z-10 relative">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="confGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="attempt" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={{ stroke: '#334155' }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={{ stroke: '#334155' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#e2e8f0', fontSize: '13px' }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Area type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={3} fill="url(#scoreGradient)" name="Score" />
                <Area type="monotone" dataKey="confidence" stroke="#22d3ee" strokeWidth={3} fill="url(#confGradient)" name="Confidence" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* No Data State */}
      {!progress?.has_data && (
        <Card className="glass-panel p-12 flex flex-col items-center gap-4 text-center">
          <AlertCircle className="w-10 h-10 text-slate-600" />
          <h3 className="text-xl font-semibold text-slate-300">No Attempts Yet</h3>
          <p className="text-slate-500 text-sm max-w-sm">Complete your first interview evaluation to start tracking your improvement trajectory.</p>
          <Button onClick={onBack} className="mt-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white border-0">
            Start Interview
          </Button>
        </Card>
      )}

      {/* Attempt History */}
      {attempts.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold px-1">Attempt History</h3>
          {attempts.map((item) => (
            <Card
              key={item.attempt_id}
              className="glass-panel p-5 hover:bg-slate-800/80 hover:border-slate-600 transition-all cursor-pointer group"
              onClick={() => setExpandedAttempt(expandedAttempt === item.attempt_id ? null : item.attempt_id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex gap-4 items-center">
                  <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                    {item.improved
                      ? <TrendingUp className="w-5 h-5 text-emerald-400" />
                      : <CheckCircle2 className="w-5 h-5 text-slate-500" />
                    }
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="font-medium text-slate-200 group-hover:text-indigo-300 transition-colors">
                      Attempt #{item.attempt_id}
                      {item.improved && <span className="ml-2 text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full font-semibold uppercase">Improved</span>}
                    </span>
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {new Date(item.timestamp).toLocaleString()}</span>
                      <span>{item.fillers} fillers</span>
                    </div>
                  </div>
                </div>
                <div className="text-right flex items-center gap-4">
                  <div className="flex flex-col">
                    <span className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-1">Score</span>
                    <span className="font-mono text-xl font-bold text-indigo-400">{Math.round(item.score * 10)}</span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="text-rose-400 hover:text-rose-300 hover:bg-rose-400/10 rounded-full"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteAttempt(item.attempt_id);
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* Expanded Details */}
              {expandedAttempt === item.attempt_id && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} className="mt-5 pt-5 border-t border-slate-700/50 space-y-4">
                  {/* Transcript */}
                  <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                    <p className="text-slate-500 text-[10px] uppercase tracking-wider mb-2 font-semibold">Transcript</p>
                    <div className="max-h-40 overflow-y-auto pr-2 custom-scrollbar">
                        <p className="text-slate-300 text-sm leading-relaxed italic">"{item.transcript}"</p>
                    </div>
                  </div>
                  {/* Feedback */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Positives
                      </h4>
                      {(item.feedback?.positives ?? []).map((p, pi) => (
                        <div key={pi} className="flex gap-2 items-start bg-emerald-500/5 border border-emerald-500/20 p-2.5 rounded-lg">
                          <div className="w-1 h-1 mt-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                          <p className="text-slate-300 text-xs leading-relaxed">{p}</p>
                        </div>
                      ))}
                    </div>
                    <div className="space-y-2">
                      <h4 className="text-[10px] font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" /> Improvements
                      </h4>
                      {(item.feedback?.improvements ?? []).map((imp, ii) => (
                        <div key={ii} className="flex gap-2 items-start bg-rose-500/5 border border-rose-500/20 p-2.5 rounded-lg">
                          <div className="w-1 h-1 mt-1.5 rounded-full bg-rose-400 flex-shrink-0" />
                          <p className="text-slate-300 text-xs leading-relaxed">{imp}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  {item.feedback?.coaching_summary && (
                    <div className="bg-indigo-500/5 border border-indigo-500/20 p-3 rounded-lg">
                      <p className="text-indigo-200 text-sm leading-relaxed">{item.feedback.coaching_summary}</p>
                    </div>
                  )}
                </motion.div>
              )}
            </Card>
          ))}
        </div>
      )}
    </motion.div>
  );
}
