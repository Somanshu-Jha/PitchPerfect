import { useState, useRef, useEffect } from 'react';
import FadeIn from './FadeIn';

import { Upload } from 'lucide-react';

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

interface RecorderSectionProps {
  onAnalysisStart: () => void;
  onAnalysisComplete: (data: any) => void;
  onScrollToResults: () => void;
  strictness?: string;
  onResumeHint?: (name: string) => void;
  onTranscriptHint?: (text: string) => void;
}

/**
 * RecorderSection – production recording UI with live transcript,
 * dynamic-height text area, and deep analysis pipeline integration.
 */
export default function RecorderSection({
  onAnalysisStart,
  onAnalysisComplete,
  onScrollToResults,
  strictness = "intermediate",
  onResumeHint,
  onTranscriptHint,
}: RecorderSectionProps) {
  const [status, setStatus] = useState<'idle' | 'recording' | 'analyzing'>('idle');
  const [transcript, setTranscript] = useState('');
  const [time, setTime] = useState(0);
  const [error, setError] = useState('');
  const [wordCount, setWordCount] = useState(0);
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resumeInputRef = useRef<HTMLInputElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  const handleResumeUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
       setResumeFile(file);
       onResumeHint?.(file.name);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setTranscript('Processing uploaded audio file...');
      setStatus('analyzing');
      onAnalysisStart();
      onScrollToResults();
      await runAnalysis([file]);
      e.target.value = '';
    }
  };

  // Recording timer
  useEffect(() => {
    if (status === 'recording') {
      timerRef.current = setInterval(() => setTime((t) => t + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [status]);

  // Update word count whenever transcript changes
  useEffect(() => {
    const words = transcript.trim().split(/\s+/).filter(Boolean);
    setWordCount(words.length);
  }, [transcript]);

  // Auto-scroll transcript to bottom during live recording
  useEffect(() => {
    if (transcriptRef.current && status === 'recording') {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript, status]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  const startRecording = async () => {
    setError('');
    setTranscript('');
    setTime(0);
    setWordCount(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      // Connect to the streaming websocket
      const ws = new WebSocket("ws://127.0.0.1:8000/student/stream");
      wsRef.current = ws;
      
      ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.status === 'live' && data.text) {
                setTranscript(data.text);
            }
        } catch(e) {}
      };

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            chunksRef.current.push(e.data);
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(e.data);
            }
        }
      };

      // Emit chunk every 500ms for live processing
      mediaRecorder.start(500);

      setStatus('recording');
    } catch (err) {
      setError('Microphone access denied. Please allow microphone and try again.');
    }
  };

  const stopAndAnalyze = async () => {
    setStatus('analyzing');
    onAnalysisStart();
    onTranscriptHint?.(transcript);
    onScrollToResults();

    if (wsRef.current) {
      try { wsRef.current.close(); } catch { }
    }

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.onstop = async () => {
        const stream = (mediaRecorderRef.current as any)?.stream as MediaStream | undefined;
        stream?.getTracks().forEach((t) => t.stop());
        await runAnalysis(chunksRef.current);
      };
      mediaRecorderRef.current.stop();
    } else {
      await runAnalysis([]);
    }
  };

  const runAnalysis = async (chunks: BlobPart[]) => {
    try {
      let result: any = null;

      if (chunks.length > 0) {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        console.log('🔥 [RecorderSection] Audio blob ready:', blob.size, 'bytes');
        const fd = new FormData();
        fd.append('file', blob, 'interview.webm');
        if (resumeFile) {
            fd.append('resume', resumeFile, resumeFile.name);
        }
        fd.append('strictness', strictness);
        
        const userId = localStorage.getItem('auth_email') || 'local_demo';
        fd.append('user_id', userId);

        console.log(`🔥 [RecorderSection] Sending POST to /student/evaluate (Mode: ${strictness})...`);
        const res = await fetch('http://127.0.0.1:8000/student/evaluate', {
          method: 'POST',
          body: fd,
        });
        console.log('🔥 [RecorderSection] Response status:', res.status);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        console.log('🔥 [RecorderSection] Response JSON:', json);
        const payload = json?.payload ?? json?.data ?? json;

        const rawScore = payload?.scores?.overall_score ?? 6;
        const score = Math.round(rawScore <= 10 ? rawScore * 10 : rawScore);

        const confMap: Record<string, number> = { low: 30, medium: 60, high: 90 };
        const confRaw = (payload?.scores?.confidence ?? 'medium') as string;
        let confidence = payload?.confidence?.dynamic_confidence;
        if (typeof confidence !== 'number') {
           confidence = confMap[confRaw.toLowerCase()] ?? 60;
        } else {
           confidence = Math.round(confidence);
        }

        // Extract full transcript from backend
        const backendTranscript = payload?.refined_transcript || payload?.raw_transcript || transcript || '';

        // Extract feedback
        const fb = payload?.feedback;
        let feedbackStr = '';
        if (fb && typeof fb === 'object' && !Array.isArray(fb)) {
          feedbackStr = [
            ...(fb.positives ?? []).map((s: any) => `✔ ${s.text || s.sentence || s}`),
            ...(fb.improvements ?? []).map((s: any) => `✦ ${s.text || s.sentence || s}`),
          ].join('\n');
        } else if (Array.isArray(fb)) {
          feedbackStr = fb.map((s: any) => `✦ ${s.text || s.sentence || s}`).join('\n');
        } else if (typeof fb === 'string') {
          feedbackStr = fb;
        }

        // Extract audio reasoning
        const audioReasoning = fb?.audio_reasoning || payload?.audio_features?.reasoning || {};
        
        // Extract detailed DL scores
        const dlMetrics = payload?.scores?.details?.dl_metrics || {};
        
        // Extract audio features for display
        const audioFeatures = payload?.audio_features || {};
        
        // Extract 16-dimension rubric breakdown from backend
        const rubricBreakdown = payload?.rubric_breakdown || [];
        
        // Extract structured positives and improvements
        const positives = (fb?.positives ?? []).map((s: any) => typeof s === 'string' ? s : (s?.text || s?.sentence || String(s)));
        const improvements = (fb?.improvements ?? []).map((s: any) => typeof s === 'string' ? s : (s?.text || s?.sentence || String(s)));
        
        // Extract suggestions / sentence variations from feedback
        const suggestions = (fb?.suggestions ?? []).map((s: any) => typeof s === 'string' ? s : (s?.text || String(s)));
        
        // Extract score deduction reasoning
        const scoreDeductionReason = payload?.rubric_breakdown
          ?.filter((d: any) => d.raw_score < 7 && d.reasoning)
          .map((d: any) => `${d.label}: ${d.reasoning}`)
          .join(' | ') || '';

        // Extract resume alignment data (matched/missed resume items vs pitch)
        const resumeAlignment = payload?.resume_alignment || {};
        const resumeMatched = resumeAlignment?.matched || [];
        const resumeMissed = resumeAlignment?.missed || [];

        result = { 
          score, 
          confidence, 
          feedback: feedbackStr,
          transcript: backendTranscript,
          audioReasoning,
          dlMetrics,
          audioFeatures,
          coachingSummary: fb?.coaching_summary || '',
          rubricBreakdown,
          positives,
          improvements,
          suggestions,
          scoreDeductionReason,
          rubricScore: payload?.scores?.rubric_score ?? null,
          dlRawScore: payload?.scores?.dl_raw_score ?? null,
          resumeMatched,
          resumeMissed,
        };
      } else {
        throw new Error('No audio captured.');
      }

      // Update the transcript box with the backend-returned transcript
      if (result?.transcript) {
        setTranscript(result.transcript);
      }

      onAnalysisComplete(result);
    } catch (err: any) {
      console.error('[Analysis error]', err);
      setTranscript('');
      onAnalysisComplete({
        score: 0,
        confidence: 0,
        feedback: 'Backend could not be reached. Make sure the server is running on port 8000.',
        transcript: transcript || '',
        audioReasoning: {},
        dlMetrics: {},
        audioFeatures: {},
        coachingSummary: '',
      });
    } finally {
      setStatus('idle');
    }
  };

  return (
    <section
      id="recorder"
      className="min-h-screen flex flex-col items-center justify-center py-24 px-6 md:px-12"
    >
      <FadeIn duration={600} yOffset={30}>
        <div className="text-center mb-16 space-y-4">
          <h2 className="text-3xl md:text-5xl font-black tracking-tighter text-slate-800 dark:text-white">
            RECORDING STUDIO
          </h2>
          <p className="text-slate-500 dark:text-gray-400 font-medium text-lg max-w-xl mx-auto">
            Introduce yourself just like you would in a real interview. We'll capture your transcript in real-time.
          </p>
        </div>
      </FadeIn>

      <div className="w-full max-w-4xl flex flex-col gap-10">

        <FadeIn delay={200}>
          <div className="relative group w-full">
            <div className={`absolute -inset-4 bg-accent opacity-[0.03] blur-3xl rounded-[40px] transition duration-1000 ${status === 'recording' ? 'animate-pulse opacity-[0.08]' : ''}`} />

            <div className="relative saas-card p-10 md:p-14 flex flex-col gap-10 bg-white dark:bg-black z-10 rounded-[32px] border-slate-100 dark:border-white/10 shadow-[0_20px_50px_rgba(0,0,0,0.03)] focus-within:shadow-[0_20px_50px_rgba(37,99,235,0.06)] transition-all">

              <div className="flex items-center justify-between pb-8 border-b border-slate-100 dark:border-white/10">
                <div className={`flex items-center gap-3 px-4 py-2 rounded-full border shadow-sm transition-all duration-300 ${status === 'recording' ? 'bg-red-50 border-red-200 text-error' :
                    status === 'analyzing' ? 'bg-orange-50 border-orange-200 text-orange-600' :
                      'bg-slate-50 dark:bg-black border-slate-200 dark:border-white/20 text-slate-500 dark:text-gray-400 font-semibold'
                  }`}>
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${status === 'recording' ? 'bg-error animate-pulse' :
                        status === 'analyzing' ? 'bg-orange-500 animate-spin border-2 border-t-transparent' :
                          'bg-slate-300'
                      }`}
                  />
                  <span className="text-[12px] font-black tracking-widest uppercase">
                    {status === 'recording' ? 'LIVE RECORDING' :
                      status === 'analyzing' ? 'ANALYZING VOICE' :
                        'SYSTEM STANDBY'}
                  </span>
                </div>

                <div className="flex items-center gap-4">
                  {/* Word count indicator */}
                  {(status === 'recording' || wordCount > 0) && (
                    <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
                      <span className="tabular-nums">{wordCount}</span>
                      <span>words</span>
                    </div>
                  )}
                  <div className={`font-mono text-3xl font-black tracking-tighter tabular-nums px-4 py-1.5 rounded-xl border border-transparent shadow-inner ${status === 'recording' ? 'bg-slate-50 dark:bg-black border-slate-100 dark:border-white/10 text-slate-800 dark:text-white' : 'text-slate-300'}`}>
                    {formatTime(time)}
                  </div>
                </div>
              </div>

              {/* Dynamic Transcript Box (Replaced with Wave Animation for UX) */}
              <div className="relative w-full rounded-2xl bg-[#f8fafc] border border-slate-200 dark:border-white/20 shadow-inner overflow-hidden focus-within:ring-4 focus-within:ring-accent/5 focus-within:border-accent/20 transition-all flex flex-col items-center justify-center min-h-[200px]">
                <button
                  aria-label="recorder text area shadow"
                  className="absolute inset-0 w-full h-full pointer-events-none shadow-[inset_0_2px_10px_rgba(0,0,0,0.02)]"
                />
                <style>{`
                  @keyframes ui-wave {
                    0%, 100% { height: 16px; opacity: 0.5; }
                    50% { height: 80px; opacity: 1; }
                  }
                  .wave-bar {
                    animation: ui-wave 1s ease-in-out infinite;
                    width: 8px;
                    border-radius: 4px;
                    background-color: #3b82f6; /* accent blue */
                  }
                `}</style>
                <div className="z-10 flex flex-col items-center justify-center w-full">
                  {status === 'recording' ? (
                    <div className="flex flex-col items-center gap-8">
                      <div className="flex items-center gap-2 h-24">
                        <div className="wave-bar" style={{ animationDelay: '0.0s', height: '30px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.1s', height: '60px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.2s', height: '80px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.3s', height: '50px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.4s', height: '40px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.5s', height: '70px' }} />
                        <div className="wave-bar" style={{ animationDelay: '0.6s', height: '30px' }} />
                      </div>
                    </div>
                  ) : status === 'analyzing' ? (
                     <div className="flex flex-col items-center gap-4">
                        <span className="text-accent animate-pulse font-bold text-2xl">Processing Audio & Generating Feedback...</span>
                     </div>
                  ) : (
                    <span className="text-slate-400 font-semibold text-lg">Start speaking, and we'll track your voice right here...</span>
                  )}
                </div>
              </div>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 text-error text-sm font-semibold rounded-xl text-center animate-shake">
                  {error}
                </div>
              )}
            </div>
          </div>
        </FadeIn>

        {/* Action Controls Section */}
        <FadeIn delay={400} className="flex flex-col items-center gap-8 py-6">
          <div className="relative">
            <button
              onClick={status === 'idle' ? startRecording : status === 'recording' ? stopAndAnalyze : undefined}
              disabled={status === 'analyzing'}
              className={`group relative w-28 h-28 rounded-full flex flex-col items-center justify-center transition-all duration-500 transform active:scale-90 z-20 ${status === 'idle'
                  ? 'bg-white dark:bg-black border-2 border-slate-200 dark:border-white/20 text-slate-400 hover:text-accent hover:border-accent/40 shadow-[0_10px_20px_rgba(0,0,0,0.05)] hover:shadow-[0_20px_30px_rgba(0,0,0,0.08)] hover:-translate-y-1 animate-soft-pulse'
                  : status === 'recording'
                    ? 'bg-white dark:bg-black text-error border-2 border-error shadow-[0_0_0_10px_rgba(239,68,68,0.1)]'
                    : 'bg-slate-50 dark:bg-black text-slate-300 border-2 border-slate-200 dark:border-white/20 scale-95 cursor-wait'
                }`}
            >
              <div className={`flex items-center justify-center w-24 h-24 rounded-full transition-all duration-500 overflow-hidden ${status === 'recording' ? 'bg-red-50 animate-recording-ring' : 'bg-slate-50 dark:bg-black group-hover:bg-blue-50'}`}>
                {status === 'recording' ? (
                  <div className="w-8 h-8 bg-error rounded-md" />
                ) : status === 'analyzing' ? (
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="animate-spin text-accent">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                ) : (
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" x2="12" y1="19" y2="22" />
                  </svg>
                )}
              </div>
            </button>
          </div>

          <div className="text-center">
            {status === 'idle' && (
              <div className="flex flex-col items-center gap-4">
                <p className="text-sm font-black uppercase tracking-[0.2em] text-slate-400">
                  Tap to record
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,.mp3,.wav,.webm"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                
                <input
                  ref={resumeInputRef}
                  type="file"
                  accept=".pdf,.txt,.docx"
                  className="hidden"
                  onChange={handleResumeUpload}
                />

                <div className="flex flex-wrap items-center justify-center gap-3">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-bold text-slate-500 dark:text-gray-400 bg-white dark:bg-black border border-slate-200 dark:border-white/20 rounded-lg hover:bg-slate-50 hover:text-accent hover:border-accent/40 transition-colors shadow-sm"
                  >
                    <Upload className="w-4 h-4" />
                    Upload Audio
                  </button>
                  
                  <button
                    onClick={() => resumeInputRef.current?.click()}
                    className={`flex items-center gap-2 px-4 py-2 text-sm font-bold border rounded-lg transition-colors shadow-sm ${
                      resumeFile 
                        ? 'bg-blue-50 text-blue-600 border-blue-200' 
                        : 'text-slate-500 dark:text-gray-400 bg-white dark:bg-black border-slate-200 dark:border-white/20 hover:bg-slate-50 hover:text-accent hover:border-accent/40'
                    }`}
                  >
                    <Upload className="w-4 h-4" />
                    {resumeFile ? 'Resume Attached ✓' : 'Attach Resume (Optional)'}
                  </button>
                </div>
              </div>
            )}
            {status === 'recording' && (
              <p className="text-sm font-black uppercase tracking-[0.2em] text-error animate-pulse">
                Click to stop & analyze
              </p>
            )}
            {status === 'analyzing' && (
              <p className="text-sm font-black uppercase tracking-[0.2em] text-accent animate-pulse">
                Analyzing Insights...
              </p>
            )}
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
