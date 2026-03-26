import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Play, History } from 'lucide-react';
import { motion } from 'framer-motion';

export default function LandingPage({ onStart, onHistory }: { onStart: () => void, onHistory: () => void }) {
  return (
    <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-xl">
      <Card className="glass p-12 flex flex-col items-center text-center space-y-10 rounded-[2rem] border-slate-800 shadow-2xl relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent pointer-events-none" />
        
        <div className="space-y-5 z-10 mt-4">
          <div className="inline-flex px-4 py-1.5 rounded-full bg-slate-800/80 text-indigo-300 text-xs font-semibold uppercase tracking-widest mb-2 border border-slate-700 shadow-inner">Introlytics AI</div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white font-sans">Interview Evaluator</h1>
          <p className="text-slate-400 text-lg md:text-xl font-light max-w-[20rem] mx-auto leading-relaxed">Elevate your interview performance with high-precision AI analysis.</p>
        </div>

        <div className="flex flex-col sm:flex-row w-full gap-4 z-10 pt-6">
          <Button onClick={onStart} size="lg" className="flex-1 h-14 text-base font-semibold rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-lg shadow-indigo-500/20 border-0 transition-all">
            <Play className="mr-2 h-5 w-5" /> Start Interview
          </Button>
          <Button onClick={onHistory} variant="secondary" size="lg" className="flex-1 h-14 text-base font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 shadow-inner transition-all">
            <History className="mr-2 h-5 w-5 text-slate-400" /> View History
          </Button>
        </div>
      </Card>
    </motion.div>
  );
}
