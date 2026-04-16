import { useState } from 'react';
import { Shield, Key, Settings, Lock } from 'lucide-react';
import FadeIn from './FadeIn';

interface DeveloperControlsProps {
  currentStrictness: string;
  onStrictnessChange: (mode: string) => void;
}

export default function DeveloperControls({
  currentStrictness,
  onStrictnessChange,
}: DeveloperControlsProps) {
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [passkey, setPasskey] = useState('');
  const [error, setError] = useState(false);

  const handleUnlock = async (e: React.FormEvent) => {
    e.preventDefault();
    if (passkey === '@Somanshujha1000') {
      setIsUnlocked(true);
      setError(false);
      // Fetch global strictness immediately on unlock
      try {
        const res = await fetch('http://localhost:8000/auth/admin/config');
        const data = await res.json();
        if (data.success && data.universal_strictness) {
           onStrictnessChange(data.universal_strictness);
        }
      } catch (err) {}
    } else {
      setError(true);
      setTimeout(() => setError(false), 2000);
    }
  };
  
  const handleUniversalChange = async (mode: string) => {
    // Optimistic UI update
    onStrictnessChange(mode);
    try {
      await fetch('http://localhost:8000/auth/admin/config', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ strictness: mode })
      });
    } catch (err) {
      console.error("Failed to set universal strictness");
    }
  };

  const modes = [
    { id: 'beginner', label: 'Beginner', desc: 'Encouraging & Soft', color: 'bg-green-500' },
    { id: 'intermediate', label: 'Intermediate', desc: 'Balanced Corporate', color: 'bg-blue-500' },
    { id: 'advance', label: 'Advance', desc: 'Rigorous & Detailed', color: 'bg-purple-500' },
    { id: 'extreme', label: 'Extreme (FAANG)', desc: 'Zero Tolerance', color: 'bg-red-600' },
  ];

  if (!isUnlocked) {
    return (
      <div className="max-w-md mx-auto my-12 pointer-events-auto">
        <div className="saas-card p-8 bg-white/80 backdrop-blur-xl border-slate-200 dark:border-white/20 shadow-xl rounded-3xl">
          <div className="flex flex-col items-center gap-6">
            <div className="w-16 h-16 rounded-2xl bg-slate-50 dark:bg-black flex items-center justify-center border border-slate-100 dark:border-white/10">
              <Lock className="w-8 h-8 text-slate-400" />
            </div>
            <div className="text-center space-y-2">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white">Developer Access</h3>
              <p className="text-sm text-slate-500 dark:text-gray-400">Enter Passkey to manage Judgement Engine</p>
            </div>
            <form onSubmit={handleUnlock} className="w-full space-y-4">
              <div className="relative">
                <input
                  type="password"
                  value={passkey}
                  onChange={(e) => setPasskey(e.target.value)}
                  placeholder="Enter Passkey..."
                  className={`w-full px-5 py-3 rounded-xl border-2 bg-slate-50/50 outline-none transition-all ${
                    error ? 'border-red-400 animate-shake' : 'border-slate-100 dark:border-white/10 focus:border-accent/30'
                  }`}
                />
                <Key className="absolute right-4 top-3.5 w-5 h-5 text-slate-300" />
              </div>
              <button
                type="submit"
                className="w-full py-3 bg-slate-900 text-white rounded-xl font-bold hover:bg-black transition-all shadow-lg active:scale-95"
              >
                Unlock Controls
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  return (
    <FadeIn duration={500}>
      <div className="max-w-4xl mx-auto my-12 pointer-events-auto">
        <div className="saas-card p-10 bg-white dark:bg-black border-slate-200 dark:border-white/20 shadow-2xl rounded-[32px] overflow-hidden relative">
          {/* Header */}
          <div className="flex items-center justify-between mb-10 pb-6 border-b border-slate-100 dark:border-white/10">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-orange-50 flex items-center justify-center border border-orange-100">
                <Settings className="w-6 h-6 text-orange-500" />
              </div>
              <div>
                <h3 className="font-black text-xl text-slate-800 dark:text-white tracking-tight uppercase">Judgement Engine Calibration</h3>
                <p className="text-xs font-bold text-slate-400 tracking-widest uppercase">Backend Team Control Panel</p>
              </div>
            </div>
            <div className="px-3 py-1 rounded-full bg-slate-100 dark:bg-zinc-900 text-[10px] font-black tracking-widest text-slate-500 dark:text-gray-400 uppercase">
              Secure Session Active
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {modes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => handleUniversalChange(mode.id)}
                className={`relative p-6 rounded-2xl border-2 transition-all group flex flex-col gap-3 text-left ${
                  currentStrictness === mode.id
                    ? `border-accent bg-accent/5 ring-4 ring-accent/5`
                    : 'border-slate-50 bg-slate-50/30 hover:border-slate-200 hover:bg-white'
                }`}
              >
                <div className={`w-3 h-3 rounded-full ${mode.color}`} />
                <div>
                  <h4 className={`font-bold transition-all ${
                    currentStrictness === mode.id ? 'text-accent' : 'text-slate-700 dark:text-gray-200'
                  }`}>
                    {mode.label}
                  </h4>
                  <p className="text-[11px] font-medium text-slate-400 group-hover:text-slate-500">
                    {mode.desc}
                  </p>
                </div>
                {currentStrictness === mode.id && (
                  <div className="absolute top-4 right-4 animate-bounce">
                     <Shield className="w-4 h-4 text-accent fill-accent/10" />
                  </div>
                )}
              </button>
            ))}
          </div>

          <div className="mt-8 p-4 bg-slate-50 dark:bg-black rounded-xl border border-slate-100 dark:border-white/10">
            <p className="text-[11px] text-slate-500 dark:text-gray-400 font-medium leading-relaxed">
              <span className="font-bold text-slate-700 dark:text-gray-200">UNIVERSAL DEPLOYMENT CONTROL:</span> Any changes made here are permanently saved to the backend database. 
              <span className="font-bold text-red-600"> Extreme Mode</span> is for FAANG-level validation. This setting will immediately apply to ALL users globally across the platform.
            </p>
          </div>
        </div>
      </div>
    </FadeIn>
  );
}
