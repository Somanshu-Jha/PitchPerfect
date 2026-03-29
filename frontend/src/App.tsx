import { useState } from 'react';
import LandingPage from './components/screens/LandingPage';
import InterviewScreen from './components/screens/InterviewScreen';
import ProcessingState from './components/screens/ProcessingState';
import ResultScreen from './components/screens/ResultScreen';
import HistoryDashboard from './components/screens/HistoryDashboard';
import LoginPage from './components/screens/LoginPage';
import SignupPage from './components/screens/SignupPage';
import { AuthProvider, useAuth } from './contexts/AuthContext';

export type AppScreen = 'landing' | 'interview' | 'processing' | 'result' | 'history' | 'login' | 'signup';

function AppContent() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('landing');
  const [resultData, setResultData] = useState<any>(null);
  const { user } = useAuth();

  const handleFinishRecording = async (audioBlob: Blob) => {
    setCurrentScreen('processing');
    const formData = new FormData();
    const filename = (audioBlob as File).name || 'interview.webm';
    formData.append('file', audioBlob, filename);
    if (user) {
      formData.append('user_id', user.user_id);
    }

    try {
      console.log(`[API Request] Sending FormData to http://127.0.0.1:5000/student/evaluate`);
      const res = await fetch('http://127.0.0.1:5000/student/evaluate', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Backend returned HTTP ${res.status}`);
      const json = await res.json();
      console.log('[API Response]', json);
      const payload = json?.data ?? json;
      setResultData(payload);
    } catch (err) {
      console.error('[Evaluation Error] Failed to fetch from backend:', err);
      // Fallback
      setResultData({
        scores: { overall_score: 6, confidence: 'medium' },
        feedback: {
          positives: ["You attempted the interview."],
          improvements: ["Backend could not be reached. Check the server is running on port 5000."]
        },
        raw_transcript: '',
        refined_transcript: ''
      });
    } finally {
      setCurrentScreen('result');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex flex-col font-sans relative overflow-x-hidden">
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-indigo-600/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600/10 blur-[120px] rounded-full pointer-events-none" />
      
      <main className="flex-1 flex flex-col items-center justify-center p-4 md:p-8 z-10 w-full">
        {currentScreen === 'landing' && (
          <LandingPage 
            onStart={() => setCurrentScreen('interview')} 
            onHistory={() => setCurrentScreen('history')}
            onLogin={() => setCurrentScreen('login')}
            onSignup={() => setCurrentScreen('signup')}
          />
        )}
        {currentScreen === 'login' && <LoginPage onBack={() => setCurrentScreen('landing')} onLoginSuccess={() => setCurrentScreen('landing')} onGoSignup={() => setCurrentScreen('signup')} />}
        {currentScreen === 'signup' && <SignupPage onBack={() => setCurrentScreen('landing')} onSignupSuccess={() => setCurrentScreen('landing')} onGoLogin={() => setCurrentScreen('login')} />}
        {currentScreen === 'interview' && <InterviewScreen onFinish={handleFinishRecording} />}
        {currentScreen === 'processing' && <ProcessingState />}
        {currentScreen === 'result' && <ResultScreen resultData={resultData} onRetry={() => setCurrentScreen('landing')} onHistory={() => setCurrentScreen('history')} />}
        {currentScreen === 'history' && <HistoryDashboard onBack={() => setCurrentScreen('landing')} />}
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
