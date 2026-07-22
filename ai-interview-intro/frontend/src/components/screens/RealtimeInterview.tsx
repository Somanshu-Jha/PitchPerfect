import React, { useState, useRef, useEffect, Suspense, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { Avatar3D } from '../Avatar3D';
import { CandidateObserver, loadMediaPipeFaceLandmarker } from '../simulation/CandidateObserver';
import { CalibrationOverlay } from '../simulation/CalibrationOverlay';
import type { CalibrationModel, ProctorEvent, ProctorPolicyConfig } from '../simulation/ProctorTypes';
import InterviewReport from './InterviewReport';
import { Mic, PhoneOff, Zap, Shield, Loader2, Video } from 'lucide-react';
import { wsUrl } from '../../config/api.config';

// ── Types ──
type Step = 'setup' | 'calibration' | 'interview' | 'report';
interface Turn { role: 'assistant' | 'user'; content: string; }

const MODES = [
  { id: 'self_intro', name: 'Self Introduction', icon: '👤' },
  { id: 'hr', name: 'HR Round', icon: '💼' },
  { id: 'technical', name: 'Technical Round', icon: '⚙️' },
  { id: 'behavioral', name: 'Behavioral (STAR)', icon: '🧠' },
];

export default function RealtimeInterview({ onBack }: { onBack: () => void }) {
  const [step, setStep] = useState<Step>('setup');
  const [selectedMode, setSelectedMode] = useState(MODES[1]);
  const [difficulty, setDifficulty] = useState('intermediate');

  // WS & Interview State
  const wsRef = useRef<WebSocket | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const [history, setHistory] = useState<Turn[]>([]);
  const historyRef = useRef<Turn[]>([]);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState<'idle' | 'listening' | 'thinking' | 'speaking' | 'generating_report'>('idle');
  const statusRef = useRef(status);
  useEffect(() => { statusRef.current = status; }, [status]);

  // Resume & Job Role
  const [resumeText, setResumeText] = useState(localStorage.getItem('resume_analysis_context') || '');
  const [jobRole, setJobRole] = useState('Software Engineer');

  // Audio / VAD
  const recognitionRef = useRef<any>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceTimerRef = useRef<any>(null);
  const globalSilenceTimerRef = useRef<any>(null);
  const lastTranscriptRef = useRef('');

  // Webcam
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [mouthOpenness, setMouthOpenness] = useState(0);

  // Calibration & Proctor
  const [calibration, setCalibration] = useState<CalibrationModel | null>(null);
  const [policy, setPolicy] = useState<ProctorPolicyConfig>({ dwellMultiplier: 1.0 });
  const calibFaceLandmarkerRef = useRef<any>(null);

  // Turn sequencing (overlap/repeat fix)
  const turnIdRef = useRef(0);
  const awaitingTurnRef = useRef(false);
  const coveredTopicsRef = useRef<string[]>([]);

  // Live score & integrity
  const [liveScore, setLiveScore] = useState<{ score: number; tip: string; avg: number; count: number } | null>(null);
  const [integrity, setIntegrity] = useState(100);
  const scoreSumRef = useRef(0);
  const scoreCountRef = useRef(0);

  // Report
  const [reportData, setReportData] = useState<any>(null);
  const [terminatedReason, setTerminatedReason] = useState<string | null>(null);

  useEffect(() => { historyRef.current = history; }, [history]);

  useEffect(() => {
    if (cameraStream && videoRef.current) videoRef.current.srcObject = cameraStream;
  }, [cameraStream, step]);

  // ── 1. Setup → Calibration ──
  const startInterview = async () => {
    setCameraError(null); setConnectionError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      setCameraStream(stream); setCameraActive(true);
    } catch (e: any) {
      setCameraError(e.message || 'Camera/mic access denied.');
      return;
    }
    // Pre-load FaceLandmarker for calibration
    loadMediaPipeFaceLandmarker().then(fl => { calibFaceLandmarkerRef.current = fl; });
    setStep('calibration');
  };

  const enterRoom = useCallback((model: CalibrationModel | null) => {
    setCalibration(model);
    setStep('interview');
    initWebSocket();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 2. WebSocket ──
  const initWebSocket = () => {
    setConnectionError(null);
    let socket: WebSocket;
    try { socket = new WebSocket(wsUrl('/interview/ws/stream')); }
    catch (e: any) { setConnectionError(`WS init failed: ${e.message}`); return; }

    socket.onopen = () => {
      setStatus('listening');
      startListening();
      const pingInterval = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'ping' }));
        else clearInterval(pingInterval);
      }, 10000);
      socket.addEventListener('close', () => clearInterval(pingInterval));
      socket.send(JSON.stringify({
        type: 'vad_pause', history: [], user_response: '', difficulty,
        mode: selectedMode.id, turn_id: 0, covered_topics: [],
        use_fast_mode: localStorage.getItem('selected_llm') !== 'nvidia' && localStorage.getItem('think_mode') !== 'true',
        selected_llm: localStorage.getItem('think_mode') === 'true' ? 'kimi' : (localStorage.getItem('selected_llm') || 'nvidia'),
        resume_text: resumeText, job_role: jobRole, is_kickoff: true,
      }));
    };

    socket.onmessage = async (event) => {
      if (typeof event.data !== 'string') { playAudioStream(event.data); return; }
      const msg = JSON.parse(event.data);
      if (msg.type === 'status') {
        setStatus(msg.message as any);
      } else if (msg.type === 'session_config') {
        if (msg.policy) setPolicy({ dwellMultiplier: msg.policy.dwell_multiplier ?? 1.0 });
      } else if (msg.type === 'turn_result') {
        // Sequencing: ignore stale turns (but always accept turn_id 0 = kickoff)
        if (msg.turn_id !== undefined && msg.turn_id !== 0 && msg.turn_id !== turnIdRef.current) return;
        awaitingTurnRef.current = false;
        const newQ = msg.data?.next_question || msg.data?.feedback || '';
        if (newQ) setHistory(prev => [...prev, { role: 'assistant', content: newQ }]);
        if (msg.data?.topic) coveredTopicsRef.current = [...coveredTopicsRef.current, msg.data.topic];
        if (typeof msg.data?.integrity_score === 'number') setIntegrity(msg.data.integrity_score);
        setTranscript(''); lastTranscriptRef.current = '';
      } else if (msg.type === 'answer_score') {
        const d = msg.data;
        scoreSumRef.current += d.score; scoreCountRef.current += 1;
        setLiveScore({ score: d.score, tip: d.one_line_tip, avg: scoreSumRef.current / scoreCountRef.current, count: scoreCountRef.current });
      } else if (msg.type === 'proctor_action') {
        // HR verbal warnings are delivered as audio by the backend; nothing to do here
      } else if (msg.type === 'session_terminated') {
        setTerminatedReason(msg.reason);
        stopListening();
      } else if (msg.type === 'final_report') {
        setReportData(msg.data);
        setStatus('idle');
        setStep('report');
      } else if (msg.type === 'error') {
        awaitingTurnRef.current = false;
        setStatus('listening'); startListening();
      }
    };

    socket.onerror = () => setConnectionError('Failed to connect to AI server.');
    socket.onclose = (e) => {
      if (!e.wasClean) setConnectionError('WebSocket connection lost.');
      if (wsRef.current === socket) wsRef.current = null;
    };
    wsRef.current = socket;
  };

  // ── 3. Audio Playback ──
  const playAudioStream = async (blob: Blob) => {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    const ctx = audioCtxRef.current;
    try {
      const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
      if (sourceNodeRef.current) { try { sourceNodeRef.current.stop(); } catch {} sourceNodeRef.current.disconnect(); }
      const source = ctx.createBufferSource();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.buffer = decoded;
      source.connect(analyser); analyser.connect(ctx.destination);
      analyserRef.current = analyser; sourceNodeRef.current = source;
      source.start();
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLips = () => {
        analyser.getByteFrequencyData(dataArray);
        setMouthOpenness(Math.min(1.0, dataArray.reduce((s, v) => s + v, 0) / dataArray.length / 100));
        requestAnimationFrame(updateLips);
      };
      updateLips();
      source.onended = () => { if (sourceNodeRef.current === source) { setMouthOpenness(0); setStatus('listening'); } };
    } catch { setStatus('listening'); }
  };

  // ── 4. VAD ──
  const startListening = () => {
    if (recognitionRef.current) return;
    const SpeechRec = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) { setConnectionError('Speech Recognition not supported. Use Chrome or Edge.'); return; }
    const rec = new SpeechRec();
    rec.continuous = true; rec.interimResults = true; rec.lang = 'en-US';
    rec.onresult = (e: any) => {
      let current = '';
      for (let i = 0; i < e.results.length; i++) current += e.results[i][0].transcript;
      setTranscript(current.trim()); lastTranscriptRef.current = current.trim();
      if (globalSilenceTimerRef.current) { clearTimeout(globalSilenceTimerRef.current); globalSilenceTimerRef.current = null; }
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(() => triggerVadPause(false), 6000);
    };
    rec.onerror = (e: any) => { if (e.error === 'not-allowed') setCameraError('Microphone permission blocked.'); };
    rec.onend = () => { if (statusRef.current === 'listening') { try { rec.start(); } catch {} } else { recognitionRef.current = null; } };
    recognitionRef.current = rec; rec.start();
    globalSilenceTimerRef.current = setTimeout(() => triggerVadPause(true), 12000);
  };

  const stopListening = () => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (globalSilenceTimerRef.current) clearTimeout(globalSilenceTimerRef.current);
    if (recognitionRef.current) { recognitionRef.current.onend = null; recognitionRef.current.stop(); recognitionRef.current = null; }
  };

  const triggerVadPause = (isSilence = false) => {
    if (!lastTranscriptRef.current && !isSilence) return;
    if (awaitingTurnRef.current) return; // sequencing lock
    awaitingTurnRef.current = true;
    turnIdRef.current += 1;
    stopListening(); setStatus('thinking');
    const userResponse = lastTranscriptRef.current || '[SYSTEM: The candidate was completely silent.]';
    setHistory(prev => {
      const newHistory = [...prev, { role: 'user' as const, content: userResponse }];
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'vad_pause', history: prev, user_response: userResponse,
          difficulty, mode: selectedMode.id, turn_id: turnIdRef.current,
          covered_topics: coveredTopicsRef.current,
          use_fast_mode: localStorage.getItem('selected_llm') !== 'nvidia' && localStorage.getItem('think_mode') !== 'true',
          selected_llm: localStorage.getItem('think_mode') === 'true' ? 'kimi' : (localStorage.getItem('selected_llm') || 'nvidia'),
          resume_text: resumeText, job_role: jobRole,
        }));
      }
      return newHistory;
    });
  };

  useEffect(() => { if (status === 'listening') startListening(); else stopListening(); }, [status]); // eslint-disable-line

  // ── Proctor event sender ──
  const sendProctorEvent = useCallback((e: ProctorEvent) => {
    const s = wsRef.current;
    if (s && s.readyState === WebSocket.OPEN) {
      s.send(JSON.stringify({
        type: 'proctor_event', event_type: e.eventType, confidence: e.confidence,
        ts_ms: e.tsMs, candidate_speaking: !!lastTranscriptRef.current, meta: e.meta ?? {},
      }));
    }
  }, []);

  // ── End call ──
  const handleEndCall = () => {
    stopListening();
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'end_call' }));
      setStatus('generating_report');
    } else {
      onBack();
    }
  };

  // ── Cleanup ──
  useEffect(() => {
    return () => {
      stopListening();
      wsRef.current?.close();
      cameraStream?.getTracks().forEach(t => t.stop());
    };
  }, []); // eslint-disable-line

  // ── Render: report ──
  if (step === 'report') {
    return <InterviewReport data={reportData} terminatedReason={terminatedReason} onBack={onBack} />;
  }

  // ── Render: setup ──
  if (step === 'setup') {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-6 text-white">
        <div className="max-w-2xl w-full bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-xl">
          <button onClick={onBack} className="text-white/40 hover:text-white mb-6 text-sm flex items-center gap-2">← Back to Dashboard</button>
          <h1 className="text-3xl font-black mb-8 flex items-center gap-3"><Zap className="text-yellow-400" /> 3D HR Engine</h1>
          {cameraError && <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl mb-6 text-sm"><strong>⚠️ Camera/Mic Warning:</strong> {cameraError}</div>}
          {connectionError && <div className="bg-amber-500/10 border border-amber-500/20 text-amber-400 p-4 rounded-xl mb-6 text-sm"><strong>⚠️ Connection Warning:</strong> {connectionError}</div>}
          <div className="space-y-6">
            <div>
              <label className="text-sm font-bold text-white/50 uppercase tracking-widest mb-3 block">Difficulty</label>
              <div className="flex gap-3">
                {['beginner', 'intermediate', 'advanced', 'faang'].map(d => (
                  <button key={d} onClick={() => setDifficulty(d)}
                    className={`flex-1 py-3 rounded-xl border font-semibold capitalize transition-all ${difficulty === d ? 'bg-indigo-500/20 border-indigo-500 text-indigo-300' : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10'}`}>{d}</button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-bold text-white/50 uppercase tracking-widest mb-3 block">Target Job Role</label>
                <input type="text" value={jobRole} onChange={e => setJobRole(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:outline-none focus:border-indigo-500"
                  placeholder="e.g. Senior Frontend Engineer" />
              </div>
              <div>
                <label className="text-sm font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center justify-between">
                  <span>Resume / CV</span>
                  <label className="cursor-pointer bg-white/10 hover:bg-white/20 text-xs px-3 py-1 rounded-full transition-colors">
                    Upload File
                    <input type="file" accept=".txt" className="hidden" onChange={async e => {
                      const file = e.target.files?.[0]; if (!file) return;
                      const text = await file.text(); setResumeText(text); localStorage.setItem('resume_analysis_context', text);
                    }} />
                  </label>
                </label>
                <textarea value={resumeText} onChange={e => { setResumeText(e.target.value); localStorage.setItem('resume_analysis_context', e.target.value); }}
                  className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white h-[100px] resize-none focus:outline-none focus:border-indigo-500"
                  placeholder="Paste your key experiences here..." />
              </div>
            </div>
            <div>
              <label className="text-sm font-bold text-white/50 uppercase tracking-widest mb-3 block">Interview Mode</label>
              <div className="grid grid-cols-2 gap-3">
                {MODES.map(m => (
                  <button key={m.id} onClick={() => setSelectedMode(m)}
                    className={`p-4 rounded-xl border text-left transition-all ${selectedMode.id === m.id ? 'bg-indigo-500/20 border-indigo-500' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}>
                    <div className="text-2xl mb-2">{m.icon}</div>
                    <div className="font-bold">{m.name}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
          <button onClick={startInterview} className="w-full mt-10 py-4 bg-white text-black font-black text-lg rounded-xl hover:bg-gray-200 transition-colors">
            Enter Interview Room
          </button>
        </div>
      </div>
    );
  }

  // ── Render: calibration ──
  if (step === 'calibration') {
    return (
      <CalibrationOverlay
        faceLandmarker={calibFaceLandmarkerRef.current}
        videoRef={videoRef}
        onComplete={model => enterRoom(model)}
        onSkip={() => enterRoom(null)}
      />
    );
  }

  // ── Render: interview room ──
  return (
    <div className="min-h-screen bg-black flex flex-col font-sans text-white relative overflow-hidden">
      {/* Top Bar */}
      <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-black/50 backdrop-blur-md z-20">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${status === 'listening' ? 'bg-green-400 animate-pulse' : status === 'thinking' ? 'bg-yellow-400 animate-pulse' : status === 'speaking' ? 'bg-blue-400 animate-pulse' : 'bg-white/20'}`} />
          <span className="text-sm font-medium text-white/60 capitalize">{status === 'generating_report' ? 'Preparing report…' : status}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-white/40">
          <Shield className="w-3 h-3 text-emerald-400" /> {difficulty} mode
        </div>
        <button onClick={handleEndCall} className="flex items-center gap-2 bg-red-500/20 hover:bg-red-500/40 border border-red-500/30 text-red-400 px-4 py-2 rounded-xl transition-all text-sm font-semibold">
          <PhoneOff className="w-4 h-4" /> End Call
        </button>
      </div>

      {/* Generating report overlay */}
      {status === 'generating_report' && (
        <div className="absolute inset-0 z-50 bg-black/80 flex flex-col items-center justify-center gap-4">
          <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
          <p className="text-white/60 text-sm uppercase tracking-widest">Preparing your interview report…</p>
        </div>
      )}

      {/* Connection error */}
      {connectionError && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-50 bg-red-500/10 border border-red-500/20 text-red-400 px-6 py-3 rounded-xl text-sm backdrop-blur-md">
          <strong>⚠️ Connection Alert:</strong> {connectionError}
        </div>
      )}

      {/* 3D Canvas */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-cover bg-center z-[-1]" style={{ backgroundImage: "url('/office_wall_bg.png')" }} />
        <Canvas camera={{ position: [0, 0.45, 2.95], fov: 44, near: 0.1, far: 50 }} gl={{ preserveDrawingBuffer: true, antialias: true }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[2, 3, 4]} intensity={2} color="#fff1e0" castShadow />
          <pointLight position={[-3, 2, -3]} intensity={1.5} color="#d4e8ff" />
          <pointLight position={[0, -1, 2]} intensity={0.5} color="#ffffff" />
          <Suspense fallback={<Html center><div className="flex flex-col items-center gap-4"><Loader2 className="w-12 h-12 text-blue-500 animate-spin" /><p className="text-white font-semibold">Loading HR Avatar…</p></div></Html>}>
            <Avatar3D isSpeaking={status === 'speaking'} isThinking={status === 'thinking'} mouthOpenness={mouthOpenness} />
          </Suspense>
        </Canvas>
      </div>

      {/* Webcam preview */}
      <div className="absolute bottom-6 left-6 w-48 h-32 bg-black border border-white/20 rounded-xl overflow-hidden z-20 shadow-2xl">
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform -scale-x-100" />
        <div className="absolute top-2 left-2 bg-black/50 backdrop-blur-md px-2 py-1 rounded-md flex items-center gap-2">
          <Video className="w-3 h-3 text-emerald-400" />
          <span className="text-[10px] uppercase font-bold text-white/80">Proctoring Active</span>
        </div>
      </div>

      {/* Anti-cheat panel */}
      <div className="absolute top-6 left-6 w-48 bg-black/60 backdrop-blur-xl border border-white/20 rounded-xl p-4 z-20 shadow-2xl">
        <div className="text-[10px] uppercase font-bold text-white/50 mb-3 flex items-center gap-2">
          <Shield className="w-3 h-3" /> Anti-Cheat
        </div>
        <CandidateObserver
          videoRef={videoRef}
          isActive={cameraActive && step === 'interview'}
          policy={policy}
          calibration={calibration}
          candidateSpeaking={!!transcript}
          onProctorEvent={sendProctorEvent}
        />
      </div>

      {/* Live scorecard */}
      {liveScore && (
        <div className="absolute top-6 right-6 w-56 bg-black/60 backdrop-blur-xl border border-white/20 rounded-xl p-4 z-20 shadow-2xl">
          <div className="text-[10px] uppercase font-bold text-white/50 mb-2 flex items-center justify-between">
            <span>Live Score</span>
            <span className={integrity >= 80 ? 'text-emerald-400' : integrity >= 50 ? 'text-amber-400' : 'text-red-400'}>
              Integrity {integrity}
            </span>
          </div>
          <div className="flex items-end gap-2">
            <span className="text-3xl font-black text-white">{liveScore.score.toFixed(1)}</span>
            <span className="text-white/40 text-sm mb-1">/10 · avg {liveScore.avg.toFixed(1)}</span>
          </div>
          <p className="text-[11px] text-indigo-300 mt-2 leading-snug">💡 {liveScore.tip}</p>
        </div>
      )}

      {/* Status + transcript overlay */}
      <div className="absolute inset-0 z-50 pointer-events-none">
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50">
          {status === 'speaking' && (
            <div className="bg-blue-600/90 text-white px-6 py-2 rounded-full flex items-center gap-3 backdrop-blur-md animate-pulse">
              <Zap className="w-4 h-4" /><span className="font-medium">HR is speaking…</span>
            </div>
          )}
          {status === 'listening' && (
            <div className="bg-green-600/90 text-white px-6 py-2 rounded-full flex items-center gap-3 backdrop-blur-md animate-pulse">
              <Mic className="w-4 h-4 animate-bounce" /><span className="font-medium">Listening to you…</span>
            </div>
          )}
        </div>
        <div className="max-w-2xl mx-auto w-full mb-10 absolute bottom-6 left-6 right-6">
          {transcript && (
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-4 text-center">
              <p className="text-white/80 text-sm leading-relaxed">{transcript}<span className="animate-pulse ml-1 text-white/40">…</span></p>
            </div>
          )}
          {status === 'speaking' && history.length > 0 && history[history.length - 1].role === 'assistant' && !transcript && (
            <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-4 text-center mt-4">
              <p className="text-white font-medium text-[15px] italic leading-relaxed">"{history[history.length - 1].content}"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
