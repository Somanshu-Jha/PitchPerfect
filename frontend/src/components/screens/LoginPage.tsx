import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { LogIn, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';

export default function LoginPage({ onBack, onLoginSuccess, onGoSignup }: { onBack: () => void, onLoginSuccess: () => void, onGoSignup: () => void }) {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const res = await fetch('http://127.0.0.1:5000/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, password })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      
      login(data.access_token, data.user_id, data.name);
      onLoginSuccess();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-md">
      <Card className="glass p-8 flex flex-col space-y-6 rounded-[2rem] border-slate-800 shadow-2xl relative overflow-hidden">
        
        <button onClick={onBack} className="absolute top-6 left-6 text-slate-400 hover:text-white transition-colors">
          <ArrowLeft className="w-6 h-6" />
        </button>

        <div className="text-center space-y-2 mt-4">
          <h1 className="text-3xl font-bold tracking-tight text-white font-sans">Welcome Back</h1>
          <p className="text-slate-400 font-light">Sign in to your Introlytics account</p>
        </div>

        {error && <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg text-center">{error}</div>}

        <form onSubmit={handleLogin} className="space-y-4">
          <div className="space-y-2">
            <input 
              type="text" 
              placeholder="Email or Username" 
              required
              className="w-full bg-slate-900/50 border border-slate-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all font-light"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <input 
              type="password" 
              placeholder="Password" 
              required
              className="w-full bg-slate-900/50 border border-slate-700 text-white px-4 py-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all font-light"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <Button type="submit" disabled={loading} size="lg" className="w-full h-12 text-base font-semibold rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-lg border-0 transition-all">
            {loading ? "Verifying..." : <><LogIn className="mr-2 h-5 w-5" /> Sign In</>}
          </Button>
        </form>

        <p className="text-center text-sm text-slate-400 mt-4">
          Don't have an account? <button onClick={onGoSignup} className="text-indigo-400 hover:text-indigo-300 font-medium">Create one</button>
        </p>

      </Card>
    </motion.div>
  );
}
