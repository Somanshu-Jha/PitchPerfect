import { useState, useRef, useEffect } from 'react';
import HeroSection from './components/HeroSection';
import RecorderSection from './components/RecorderSection';
import ResultSection from './components/screens/ResultSection';
import Navbar from './components/Navbar';
import FeaturesSection from './components/FeaturesSection';
import AboutSection from './components/AboutSection';
import HistoryScreen from './components/screens/HistoryScreen';
import { LoginPage } from './components/animated-characters-login-page';
import DeveloperControls from './components/DeveloperControls';
/**
 * App – single-page scroll application.
 * Fully functional without breaking logic.
 * Clean SaaS typography and layout.
 */
function App() {
  const [resultData, setResultData] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [strictness, setStrictness] = useState('intermediate');

  // ── Persistent Login: Check token on app load ──
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      setIsCheckingAuth(false);
      return;
    }
    fetch('http://localhost:8000/auth/verify-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setIsLoggedIn(true);
        } else {
          localStorage.removeItem('auth_token');
          localStorage.removeItem('auth_email');
        }
      })
      .catch(() => {
        // Server offline: keep token, show login
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_email');
      })
      .finally(() => setIsCheckingAuth(false));
  }, []);

  const resultsRef = useRef<HTMLDivElement>(null);

  // Smooth-scroll to results section
  const scrollToResults = () => {
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  // Scroll to recorder section
  const scrollToRecorder = () => {
    document.getElementById('recorder')?.scrollIntoView({ behavior: 'smooth' });
  };

  // Reset for a new attempt
  const handleRetry = () => {
    setResultData(null);
    setIsAnalyzing(false);
    document.getElementById('recorder')?.scrollIntoView({ behavior: 'smooth' });
  };

  if (isCheckingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!isLoggedIn) {
    return <LoginPage onLogin={() => setIsLoggedIn(true)} />;
  }

  if (showHistory) {
    return <HistoryScreen onBack={() => setShowHistory(false)} />;
  }

  return (
    <div className="min-h-screen font-sans text-base relative">
      <Navbar onOpenHistory={() => setShowHistory(true)} />

      <HeroSection onDiveIn={scrollToRecorder} />

      <div className="divider-subtle" />

      <RecorderSection
        onAnalysisStart={() => {
          setResultData(null);
          setIsAnalyzing(true);
        }}
        onAnalysisComplete={(data) => {
          setIsAnalyzing(false);
          setResultData(data);
        }}
        onScrollToResults={scrollToResults}
        strictness={strictness}
      />

      <div className="divider-subtle" />

      <div ref={resultsRef}>
        <ResultSection
          data={resultData}
          isLoading={isAnalyzing}
          onRetry={handleRetry}
        />
      </div>

      <div className="divider-subtle" />
      <FeaturesSection onStartRecording={scrollToRecorder} />
      <AboutSection onStartRecording={scrollToRecorder} />

      <div className="bg-slate-50/50 py-12 border-t border-slate-100">
        <DeveloperControls 
          currentStrictness={strictness}
          onStrictnessChange={setStrictness}
        />
      </div>

      <footer className="bg-white border-t border-slate-100">
        {/* Main grid */}
        <div className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 md:grid-cols-4 gap-10">

          {/* Section 1 – Brand */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-md bg-accent flex items-center justify-center shrink-0">
                <span className="text-white font-black text-[11px]">PP</span>
              </div>
              <span className="font-black tracking-tight text-slate-800 text-base">PitchPerfect</span>
            </div>
            <p className="text-sm text-gray-500 leading-relaxed">
              AI-powered platform to analyze your interview introduction and improve clarity, confidence, and impact.
            </p>
          </div>

          {/* Section 2 – Product */}
          <div className="flex flex-col gap-3">
            <h4 className="text-sm font-semibold text-slate-700 tracking-wide">Product</h4>
            <ul className="flex flex-col gap-2">
              <li>
                <a
                  href="#features"
                  onClick={(e) => { e.preventDefault(); document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' }); }}
                  className="text-sm text-gray-500 hover:text-accent transition-colors duration-200 cursor-pointer"
                >
                  Features
                </a>
              </li>
              <li>
                <a
                  href="#how-it-works"
                  onClick={(e) => { e.preventDefault(); document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' }); }}
                  className="text-sm text-gray-500 hover:text-accent transition-colors duration-200 cursor-pointer"
                >
                  How it works
                </a>
              </li>
            </ul>
          </div>

          {/* Section 3 – Company */}
          <div className="flex flex-col gap-3">
            <h4 className="text-sm font-semibold text-slate-700 tracking-wide">Company</h4>
            <ul className="flex flex-col gap-2">
              <li>
                <a
                  href="#about"
                  onClick={(e) => { e.preventDefault(); document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' }); }}
                  className="text-sm text-gray-500 hover:text-accent transition-colors duration-200 cursor-pointer"
                >
                  About
                </a>
              </li>
              <li>
                <a
                  href="mailto:akash@gmail.com"
                  className="text-sm text-gray-500 hover:text-accent transition-colors duration-200"
                >
                  Contact
                </a>
              </li>
            </ul>
          </div>

          {/* Section 4 – Contact */}
          <div className="flex flex-col gap-3">
            <h4 className="text-sm font-semibold text-slate-700 tracking-wide">Contact</h4>
            <ul className="flex flex-col gap-2">
              {['somanshujha1@gmail.com', 'akashkumar15773728p@gmail.com', 'itzshonemshery@gmail.com'].map((email) => (
                <li key={email}>
                  <a
                    href={`mailto:${email}`}
                    className="text-sm text-gray-500 hover:text-accent transition-colors duration-200"
                  >
                    {email}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom row */}
        <div className="border-t border-slate-100">
          <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-xs text-gray-400">
              © {new Date().getFullYear()}PitchPerfect AI. All rights reserved.
            </p>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/akash15773728"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-gray-400 hover:text-accent transition-colors duration-200"
              >
                GitHub
              </a>
              <a
                href="https://x.com/Somanshu_Jha"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-gray-400 hover:text-accent transition-colors duration-200"
              >
                X (Twitter)
              </a>
              <a
                href="https://www.instagram.com/_shonemary_"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-gray-400 hover:text-accent transition-colors duration-200"
              >
                Instagram
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
