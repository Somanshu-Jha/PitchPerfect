import { useState } from 'react';
import LandingPage from './components/screens/LandingPage';
import InterviewScreen from './components/screens/InterviewScreen';
import ProcessingState from './components/screens/ProcessingState';
import ResultScreen from './components/screens/ResultScreen';
import HistoryDashboard from './components/screens/HistoryDashboard';

export type AppScreen = 'landing' | 'interview' | 'processing' | 'result' | 'history';

function App() {
  const [currentScreen, setCurrentScreen] = useState<AppScreen>('landing');
  const [resultData, setResultData] = useState<any>(null);

  const handleFinishRecording = async (audioBlob: Blob) => {
    setCurrentScreen('processing');
    const formData = new FormData();
    formData.append('audio', audioBlob, 'interview.webm');

    try {
      const res = await fetch('http://127.0.0.1:5000/student/evaluate', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Backend returned HTTP ${res.status}`);
      const json = await res.json();
      console.log('[API Response]', json);
      // Backend wraps result in { status, data: { ... } }
      const payload = json?.data ?? json;
      setResultData(payload);
    } catch (err) {
      console.error('[Evaluation Error]', err);
      // Fallback: shape must match what ResultScreen expects
      setResultData({
        scores: { overall_score: 6, confidence: 'medium' },
        feedback: {
          positives: ["You attempted the interview."],
          improvements: ["Backend could not be reached. Check the server is running on port 5000.", "Ensure your microphone recorded audio correctly."]
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
        {currentScreen === 'landing' && <LandingPage onStart={() => setCurrentScreen('interview')} onHistory={() => setCurrentScreen('history')} />}
        {currentScreen === 'interview' && <InterviewScreen onFinish={handleFinishRecording} />}
        {currentScreen === 'processing' && <ProcessingState />}
        {currentScreen === 'result' && <ResultScreen resultData={resultData} onRetry={() => setCurrentScreen('landing')} onHistory={() => setCurrentScreen('history')} />}
        {currentScreen === 'history' && <HistoryDashboard onBack={() => setCurrentScreen('landing')} />}
      </main>
    </div>
  );
}

export default App;
