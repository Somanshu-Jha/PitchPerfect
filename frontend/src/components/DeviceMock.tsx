import { useEffect, useState } from 'react';

/**
 * DeviceMock – Hero visual element representing transcription.
 * Highly detailed simulation of Standby -> Cursor Click -> Recording -> Typing -> Analyze -> Result.
 */
export default function DeviceMock() {
  const [step, setStep] = useState<'standby' | 'cursorMove' | 'click' | 'recording' | 'typing' | 'processing' | 'result'>('standby');
  const [displayText, setDisplayText] = useState('');
  const [cursorPos, setCursorPos] = useState({ x: '80%', y: '15%' });
  const [isClicking, setIsClicking] = useState(false);

  const fullTranscript = "Hi, I am Somanshu,Shone and Akash, a passionate developer building interactive applications...";

  useEffect(() => {
    let isMounted = true;

    const runFlow = async () => {
      if (!isMounted) return;

      // 1. STANDBY (1.2s)
      setStep('standby');
      setDisplayText('');
      setCursorPos({ x: '85%', y: '15%' });
      await new Promise(r => setTimeout(r, 1200));
      if (!isMounted) return;

      // 2. CURSOR MOVE (0.6s)
      setStep('cursorMove');
      setCursorPos({ x: '50%', y: '84.5%' });
      await new Promise(r => setTimeout(r, 600));
      if (!isMounted) return;

      // 3. CLICK (0.15s)
      setStep('click');
      setIsClicking(true);
      await new Promise(r => setTimeout(r, 150));
      setIsClicking(false);
      if (!isMounted) return;

      // 4. RECORDING (1s)
      setStep('recording');
      await new Promise(r => setTimeout(r, 1000));
      if (!isMounted) return;

      // 5. TYPING (2s)
      setStep('typing');
      for (let i = 0; i <= fullTranscript.length; i++) {
        setDisplayText(fullTranscript.slice(0, i));
        await new Promise(r => setTimeout(r, 30));
        if (!isMounted) break;
      }
      await new Promise(r => setTimeout(r, 800));
      if (!isMounted) return;

      // 6. PROCESSING (1.5s)
      setStep('processing');
      await new Promise(r => setTimeout(r, 1500));
      if (!isMounted) return;

      // 7. RESULT (4s before loop)
      setStep('result');
      await new Promise(r => setTimeout(r, 4000));

      // LOOP
      if (isMounted) runFlow();
    };

    runFlow();
    return () => { isMounted = false; };
  }, []);

  return (
    <div className="relative w-full max-w-[360px] mx-auto overflow-visible">

      {/* Inner card: overflow-hidden to correctly clip content to rounded corners */}
      <div
        className="relative z-10 w-full select-none bg-white rounded-[40px] transition-all duration-500 overflow-hidden border border-slate-100/50"
        style={{
          boxShadow: `
            0 30px 80px rgba(0,0,0,0.15),
            0 0 0 1px rgba(255,255,255,0.4),
            0 0 60px rgba(255, 220, 150, 0.15)
          `
        }}
      >

        {/* 🖱️ OS-Style System Cursor */}
        {(step === 'standby' || step === 'cursorMove' || step === 'click') && (
          <div
            className="absolute z-[100] pointer-events-none transition-all duration-500 ease-out transform-gpu"
            style={{
              left: cursorPos.x,
              top: cursorPos.y,
              transform: `translate(0, 0) ${isClicking ? 'scale(0.85)' : 'scale(1)'}`,
              opacity: step === 'standby' ? 0 : 1
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="#0f172a" stroke="white" strokeWidth="1.5">
              <path d="M3 2L21 12L13 14L11 22L3 2Z" />
            </svg>
            {isClicking && (
              <div className="absolute top-0 left-0 w-8 h-8 -ml-3 -mt-3 rounded-full bg-accent/20 animate-ping" />
            )}
          </div>
        )}

        {/* Top Light Layer (Overlay) */}
        <div className="absolute inset-0 rounded-2xl pointer-events-none bg-[linear-gradient(180deg,rgba(59,130,246,0.02),transparent)] z-0"></div>

        {/* Header Status Bar */}
        <div className="relative flex items-center justify-between border-b border-slate-50 p-6 z-10">
          <div className="flex gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-200" />
            <span className="w-1.5 h-1.5 rounded-full bg-slate-200" />
            <span className="w-1.5 h-1.5 rounded-full bg-slate-200" />
          </div>
          <div className="flex items-center gap-3">
            {(step === 'recording' || step === 'typing') && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-red-50 border border-red-100 animate-pulse">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
                <span className="text-[9px] font-black text-red-600 tracking-wider">LIVE</span>
              </div>
            )}
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full transition-all duration-300 ${step === 'recording' || step === 'typing' ? 'bg-error animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.4)]' : 'bg-slate-200'
                }`} />
              <span className="text-[10px] font-black tracking-[0.2em] uppercase text-slate-400">
                {step === 'standby' || step === 'cursorMove' || step === 'click' ? 'STANDBY' :
                  step === 'recording' || step === 'typing' ? 'RECORDING...' :
                    step === 'processing' ? 'ANALYZING' : 'READY'}
              </span>
            </div>
          </div>
        </div>

        {/* Mock Content (Animated Area) */}
        <div className="relative p-8 h-64 flex flex-col justify-center gap-4 bg-[#fcfdfe]/60 z-10 overflow-hidden">

          {(step === 'standby' || step === 'cursorMove' || step === 'click') && (
            <div className="text-center space-y-3 opacity-30 animate-in fade-in duration-500">
              <div className="w-14 h-14 rounded-full bg-slate-50 border border-slate-100 mx-auto flex items-center justify-center text-slate-300">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" />
                </svg>
              </div>
              <p className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">System Ready</p>
            </div>
          )}

          {(step === 'recording' || step === 'typing') && (
            <div className="space-y-4 animate-in fade-in duration-300">
              <div className="text-[16px] leading-[1.6] text-slate-700 font-semibold tracking-tight">
                {displayText}
                {step === 'typing' && (
                  <span className="inline-block w-[2px] h-4 bg-accent/60 ml-1.5 animate-cursor-blink align-middle" />
                )}
              </div>
              {step === 'recording' && displayText === '' && (
                <div className="flex flex-col gap-4 items-center mt-2">
                  <div className="flex gap-1 items-end h-8">
                    {[1, 2, 3, 4, 5, 4, 3, 2, 1].map((h, i) => (
                      <div
                        key={i}
                        className="w-1 bg-accent/40 rounded-full animate-waveform"
                        style={{
                          height: `${h * 20}%`,
                          animationDelay: `${i * 100}ms`
                        }}
                      />
                    ))}
                  </div>
                  <div className="flex gap-2 items-center px-1">
                    {[0, 100, 200].map((d) => (
                      <div key={d} className="w-1.5 h-1.5 rounded-full bg-red-400/80 animate-bounce" style={{ animationDelay: `${d}ms` }} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 'processing' && (
            <div className="flex flex-col items-center gap-6 animate-in zoom-in-95 duration-500">
              <div className="relative w-12 h-12">
                <div className="absolute inset-0 rounded-full border-2 border-slate-50" />
                <div className="absolute inset-0 rounded-full border-2 border-accent border-t-transparent animate-spin" />
              </div>
              <p className="text-[10px] font-black text-accent uppercase tracking-widest animate-pulse">Analyzing your response...</p>
            </div>
          )}

          {step === 'result' && (
            <div className="flex flex-col gap-6 animate-in slide-in-from-bottom-5 duration-1000">
              <div className="flex justify-between items-end border-b border-slate-50 pb-5">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.15em]">Quality Score</span>
                  <span className="text-5xl font-black tracking-tighter text-slate-800 tabular-nums leading-none">82</span>
                </div>
                <div className="flex items-end gap-1 mb-1">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.15em]">
                    Confidence Score
                  </span>
                  <span className="text-xl font-black text-accent shadow-sm px-2 bg-blue-50/50 rounded-lg">78%</span>
                </div>
              </div>
              <div className="p-3 bg-green-50/50 rounded-xl border border-green-100/50 flex gap-3 items-center">
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center text-success flex-shrink-0 border border-green-200">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                </div>
                <p className="text-[12px] font-bold text-slate-600 leading-tight">Good clarity and structure.</p>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Control Area */}
        <div className="relative p-8 border-t border-slate-50 bg-[#fefcf8] z-10 flex justify-center">
          <div
            className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 shadow-inner ${step === 'recording' || step === 'typing'
              ? 'bg-red-50 text-error border-2 border-error/40 scale-90 ring-4 ring-red-500/5'
              : isClicking
                ? 'bg-slate-50 scale-95 border border-slate-200'
                : 'bg-white border text-slate-300 shadow-[0_4px_10px_rgba(0,0,0,0.02)]'
              }`}>
            {step === 'recording' || step === 'typing' ? (
              <div className="w-4 h-4 bg-error rounded shadow-sm" />
            ) : (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            )}
          </div>
        </div>

        {/* 🔮 Corner Glow effect */}
        <div className="absolute bottom-0 right-0 w-32 h-32 bg-accent opacity-[0.03] blur-[60px] pointer-events-none" />
      </div>

      {/* --- EXACT GROUND FLARE EFFECT (Pure Circle/Ellipse Match) --- */}
      <div className="absolute left-1/2 -translate-x-1/2 top-full -mt-[1px] w-[120%] h-[120px] pointer-events-none z-0 flex flex-col items-center">

        {/* 1. The Grounding Shadow (Very dark, tight ellipse right under the card to ground it) */}
        <div className="absolute top-[0px] w-[75%] h-[10px] bg-slate-900/20 rounded-[100%] blur-[3px]" />

        {/* 2. The Vivid Blue Halo (The colored oval glow surrounding the white center) */}
        <div
          className="absolute top-[0px] w-[80%] h-[24px] rounded-[100%] blur-[6px]"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(70, 140, 255, 0.7) 0%, rgba(70, 140, 255, 0) 70%)',
          }}
        />

        {/* 3. The Pure White Center Ellipse (The crisp, glowing solid light source) */}
        <div
          className="absolute top-[1px] w-[45%] h-[8px] bg-white rounded-[100%] blur-[1px]"
          style={{
             boxShadow: '0 0 15px 4px rgba(255, 255, 255, 0.9), 0 0 30px 8px rgba(100, 150, 255, 0.6)'
          }}
        />

        {/* 4. The Soft Floor Puddle (Fading out smoothly into the background) */}
        <div
          className="absolute top-[8px] w-[110%] h-[60px] rounded-[100%] blur-[20px]"
          style={{
            background: 'radial-gradient(ellipse at center, rgba(70,140,255,0.25) 0%, transparent 70%)'
          }}
        />

      </div>

    </div>
  );
}