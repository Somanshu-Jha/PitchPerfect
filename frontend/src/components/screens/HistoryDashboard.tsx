import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Clock, Activity, CheckCircle2 } from 'lucide-react';

export default function HistoryDashboard({ onBack }: { onBack: () => void }) {
  const history = [
    { date: 'Today, 10:30 AM', score: 88, role: 'Software Engineer', duration: '2m 14s' },
    { date: 'Yesterday, 2:15 PM', score: 85, role: 'Software Engineer', duration: '1m 55s' },
    { date: 'Mon, 9:00 AM', score: 70, role: 'Data Scientist', duration: '3m 02s' },
  ];

  return (
    <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-4xl flex flex-col gap-8 py-8">
      
      <div className="flex items-center gap-4 border-b border-slate-800 pb-6">
        <Button variant="ghost" size="icon" onClick={onBack} className="rounded-full hover:bg-slate-800 text-slate-400">
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">Attempt History</h1>
          <p className="text-slate-400 text-sm mt-1">Review your past interview evaluations.</p>
        </div>
      </div>
      
      <div className="grid md:grid-cols-4 gap-6">
        <Card className="glass-panel p-6 flex flex-col justify-between h-32 md:col-span-1 shadow-sm">
          <div className="flex items-center gap-2 text-slate-400 mb-2">
             <Activity className="w-4 h-4 text-indigo-400" />
             <span className="text-xs uppercase tracking-wider font-semibold">Avg Score</span>
          </div>
          <span className="text-4xl font-bold text-white">81</span>
        </Card>
        
        <Card className="glass-panel p-6 flex flex-col justify-between h-32 md:col-span-3 overflow-hidden relative shadow-sm">
          <div className="flex items-center justify-between z-10">
            <span className="text-slate-400 text-xs uppercase tracking-wider font-semibold">Performance Trend</span>
          </div>
          <div className="flex-1 w-full flex items-end justify-between px-2 pt-4 z-10">
            {[60, 75, 70, 85, 88].map((score, i) => (
              <div key={i} className="flex flex-col items-center gap-2 group w-12">
                <div className="w-full bg-gradient-to-t from-indigo-600/20 to-indigo-500 rounded-t-sm transition-all duration-500 group-hover:to-purple-400" style={{ height: `${score}%` }} />
                <span className="text-[10px] text-slate-500 font-mono">{score}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="space-y-3">
        {history.map((item, i) => (
          <Card key={i} className="glass-panel p-5 flex items-center justify-between hover:bg-slate-800/80 hover:border-slate-600 transition-all cursor-pointer group">
            <div className="flex gap-4 items-center">
              <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                 <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="font-medium text-slate-200 group-hover:text-indigo-300 transition-colors">{item.role}</span>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                   <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {item.date}</span>
                   <span>•</span>
                   <span>{item.duration}</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="text-right flex flex-col">
                <span className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-1">Score</span>
                <span className="font-mono text-xl font-bold text-indigo-400">{item.score}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </motion.div>
  );
}
