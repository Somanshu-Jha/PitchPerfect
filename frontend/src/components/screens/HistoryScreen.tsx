import { useEffect, useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts';
import { ArrowLeft, History, TrendingUp, Award, Calendar, Clock } from 'lucide-react';

interface HistoryScreenProps {
  onBack: () => void;
}

export default function HistoryScreen({ onBack }: HistoryScreenProps) {
  const [data, setData] = useState<any[]>([]);
  const [rawAttempts, setRawAttempts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ avg_score: 0, avg_conf: 0, total: 0, best: 0, trend: 0 });

  useEffect(() => {
    async function fetchHistory() {
      try {
        const userId = localStorage.getItem('auth_email') || 'local_demo';
        const res = await fetch(`http://localhost:8000/student/history/${encodeURIComponent(userId)}?days=30`);
        const json = await res.json();
        if (json.status === 'success' && json.data.length > 0) {
          const attempts = json.data; // newest-first from API
          setRawAttempts(attempts);

          const chronological = [...attempts].reverse(); // oldest → newest
          const chartData = chronological.map((item: any, idx: number) => ({
            name: `#${idx + 1}`,
            score: Math.round(item.scores?.overall * 10 || 0),
            confidence: Math.round(item.scores?.confidence || 0),
            date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            fullDate: new Date(item.date).toLocaleString(),
          }));
          setData(chartData);

          // Compute stats
          const scores = attempts.map((a: any) => a.scores?.overall || 0);
          const confs = attempts.map((a: any) => a.scores?.confidence || 0);
          const avg_score = scores.length ? scores.reduce((a: number, b: number) => a + b, 0) / scores.length : 0;
          const avg_conf = confs.length ? confs.reduce((a: number, b: number) => a + b, 0) / confs.length : 0;
          const best = scores.length ? Math.max(...scores) : 0;
          // Trend: newest score (index 0) vs oldest score (last index)
          const trend = scores.length >= 2 ? scores[0] - scores[scores.length - 1] : 0;
          setStats({
            avg_score: Math.round(avg_score * 10),
            avg_conf: Math.round(avg_conf),
            total: attempts.length,
            best: Math.round(best * 10),
            trend: Math.round(trend * 10)
          });
        }
      } catch (err) {
        console.error('History fetch error', err);
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, []);

  const handleExport = () => {
    const userId = localStorage.getItem('auth_email') || 'local_demo';
    window.open(`http://localhost:8000/student/export/${encodeURIComponent(userId)}`, '_blank');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50/30 p-6 md:p-12 font-sans">
      {/* Navigation & Export */}
      <div className="flex items-center justify-between xl:max-w-7xl mx-auto mb-8">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all text-slate-700 font-semibold text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </button>
        <button
          onClick={handleExport}
          className="flex items-center gap-2 px-5 py-2.5 bg-emerald-50 border border-emerald-200 text-emerald-600 rounded-xl shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all font-bold text-sm"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Export to Excel
        </button>
      </div>

      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-200">
              <History className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-black text-slate-800 tracking-tight">Performance History</h1>
              <p className="text-slate-500 font-medium mt-1">Last 30 days • Track your progress over time</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400 font-medium">
            <Calendar className="w-3.5 h-3.5" />
            <span>{new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</span>
          </div>
        </div>

        {loading ? (
          <div className="flex h-64 items-center justify-center">
            <div className="flex flex-col items-center gap-4">
              <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-semibold text-slate-400 tracking-wide">Loading history...</span>
            </div>
          </div>
        ) : data.length === 0 ? (
          <div className="bg-white p-16 text-center rounded-3xl border border-slate-200 shadow-sm">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-slate-100 flex items-center justify-center">
              <History className="w-10 h-10 text-slate-300" />
            </div>
            <h2 className="text-2xl font-black text-slate-600 mb-2">No Attempts Yet</h2>
            <p className="text-slate-400 max-w-md mx-auto">Record your first AI interview to start tracking your progress!</p>
          </div>
        ) : (
          <>
            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <Award className="w-4 h-4 text-amber-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Best Score</span>
                </div>
                <span className="text-3xl font-black text-slate-800">{stats.best}<span className="text-lg text-slate-400">/100</span></span>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-indigo-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Avg Score</span>
                </div>
                <span className="text-3xl font-black text-slate-800">{stats.avg_score}<span className="text-lg text-slate-400">/100</span></span>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-violet-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Avg Conf</span>
                </div>
                <span className="text-3xl font-black text-slate-800">{stats.avg_conf}<span className="text-lg text-slate-400">%</span></span>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm">
                <div className="flex items-center gap-2 mb-2">
                  <Calendar className="w-4 h-4 text-emerald-500" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Attempts</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-black text-slate-800">{stats.total}</span>
                  {stats.trend > 0 && (
                    <span className="text-sm font-bold text-emerald-500">+{stats.trend}↑</span>
                  )}
                  {stats.trend < 0 && (
                    <span className="text-sm font-bold text-red-400">{stats.trend}↓</span>
                  )}
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Confidence Line Chart */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                <h3 className="text-lg font-bold text-slate-800 mb-1">Confidence Progression</h3>
                <p className="text-xs text-slate-400 mb-5">AI-measured delivery confidence over time</p>
                <div className="h-[280px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 20, left: -15, bottom: 5 }}>
                      <defs>
                        <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 30px rgb(0 0 0 / 0.08)', fontSize: '13px' }}
                        labelFormatter={(label) => {
                          const item = data.find(d => d.name === label);
                          return item ? `${label} — ${item.date}` : label;
                        }}
                      />
                      <Area type="monotone" dataKey="confidence" stroke="#8B5CF6" strokeWidth={2.5} fill="url(#confGrad)" dot={{ r: 4, fill: '#8B5CF6' }} activeDot={{ r: 7, stroke: '#fff', strokeWidth: 2 }} name="Confidence %" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Score Bar Chart */}
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                <h3 className="text-lg font-bold text-slate-800 mb-1">Score Performance</h3>
                <p className="text-xs text-slate-400 mb-5">Overall evaluation score per attempt</p>
                <div className="h-[280px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 10, right: 20, left: -15, bottom: 5 }}>
                      <defs>
                        <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#3B82F6" />
                          <stop offset="100%" stopColor="#6366F1" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} domain={[0, 100]} />
                      <Tooltip
                        cursor={{ fill: 'rgba(99, 102, 241, 0.05)' }}
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 30px rgb(0 0 0 / 0.08)', fontSize: '13px' }}
                        labelFormatter={(label) => {
                          const item = data.find(d => d.name === label);
                          return item ? `${label} — ${item.date}` : label;
                        }}
                      />
                      <Bar dataKey="score" fill="url(#scoreGrad)" radius={[6, 6, 0, 0]} name="Score /100" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Recent Attempts Table */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100">
                <h3 className="text-lg font-bold text-slate-800">Recent Attempts</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-slate-50/50">
                      <th className="px-6 py-3 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">#</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Date</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Score</th>
                      <th className="px-6 py-3 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {rawAttempts.slice(0, 10).map((a: any, i: number) => (
                      <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                        <td className="px-6 py-3 font-bold text-slate-600">{i + 1}</td>
                        <td className="px-6 py-3 text-slate-500 whitespace-nowrap">
                          {new Date(a.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td className="px-6 py-3">
                          <span className={`font-black text-base ${(a.scores?.overall || 0) >= 7 ? 'text-emerald-500' : (a.scores?.overall || 0) >= 4 ? 'text-amber-500' : 'text-red-400'}`}>
                            {Math.round((a.scores?.overall || 0) * 10)}
                          </span>
                          <span className="text-slate-300 text-xs">/100</span>
                        </td>
                        <td className="px-6 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-violet-400 to-violet-600"
                                style={{ width: `${Math.min(a.scores?.confidence || 0, 100)}%` }}
                              />
                            </div>
                            <span className="text-xs text-slate-400 font-medium">{Math.round(a.scores?.confidence || 0)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
