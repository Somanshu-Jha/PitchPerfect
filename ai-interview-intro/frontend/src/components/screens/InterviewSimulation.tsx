/**
 * InterviewSimulation.tsx — Cinematic AI Interview Room
 * ──────────────────────────────────────────────────────
 * Complete redesign: photorealistic interviewer, Kokoro TTS voice,
 * AudioContext lip-sync, MediaPipe candidate observation,
 * spring physics, Markov behavior engine.
 */

import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { apiUrl, apiHeaders } from '../../config/api.config';
import FadeIn from '../FadeIn';
import { InterviewAgentVisual } from '../InterviewAgentVisual';
import type { AvatarState } from '../InterviewAgentVisual';
import { Avatar3D } from '../Avatar3D';
import { Canvas } from '@react-three/fiber';
import { AvatarPhysicsEngine } from '../simulation/PhysicsEngine';
import { HRBehaviorEngine } from '../simulation/HRBehaviorEngine';
import type { CandidateSignal } from '../simulation/HRBehaviorEngine';
import { CandidateObserver } from '../simulation/CandidateObserver';
import type { CandidateMetrics } from '../simulation/CandidateObserver';
import {
  Mic, MicOff, Video, VideoOff, PhoneOff, MessageSquare,
  Shield, Zap, Clock, ChevronRight, Volume2, VolumeX,
} from 'lucide-react';

import hrAgentImg from '../../assets/hr_avatar.png';

// ── Types ─────────────────────────────────────────────────────────────────────

interface SimulationResult {
  feedback?: string;
  final_summary?: string;
  score?: number;
  strengths?: string[];
  improvements?: string[];
}

interface VocalParams {
  pitch_multiplier?: number;
  speed_multiplier?: number;
  pause_before_ms?: number;
  filler_prefix?: string;
}

interface SimulationTurnResponse extends SimulationResult {
  transcript?: string;
  next_question?: string;
  hr_response?: string;
  should_end?: boolean;
  avatar_state?: AvatarState;
  vocal_params?: VocalParams;
  topic?: string;
  behavior_cue?: string;
}

interface InterviewMode {
  id: string;
  name: string;
  description: string;
  question_count: number;
  icon?: string;
  voiceProfile?: string;
}

interface InterviewTurn {
  role: 'assistant' | 'user';
  content: string;
  topic?: string;
}

type SimStatus = 'idle' | 'tts_loading' | 'ai_speaking' | 'user_listening' | 'user_speaking' | 'processing';
type SimStep = 'mode' | 'loading' | 'interview' | 'result';

const DEFAULT_MODES: InterviewMode[] = [
  { id: 'self_intro',  name: 'Self Introduction', description: 'Open with who you are and what you bring',        question_count: 5, icon: '👤', voiceProfile: 'friendly_hr' },
  { id: 'hr',         name: 'HR Round',           description: 'Behavioral, culture-fit, motivation questions',   question_count: 6, icon: '💼', voiceProfile: 'behavioral_hr' },
  { id: 'technical',  name: 'Technical Round',    description: 'Deep-dive into your tech stack and decisions',    question_count: 5, icon: '⚙️', voiceProfile: 'faang_tech' },
  { id: 'behavioral', name: 'Behavioral (STAR)',  description: 'Situation-Task-Action-Result framework practice', question_count: 4, icon: '🧠', voiceProfile: 'strict_hr' },
];

const DIFFICULTY_CONFIG = {
  beginner:     { label: 'Beginner',     color: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10' },
  intermediate: { label: 'Intermediate', color: 'text-blue-400 border-blue-500/40 bg-blue-500/10' },
  advanced:     { label: 'Advanced',     color: 'text-violet-400 border-violet-500/40 bg-violet-500/10' },
  extreme:      { label: 'FAANG',        color: 'text-red-400 border-red-500/40 bg-red-500/10' },
};

// ── Main Component ────────────────────────────────────────────────────────────

export default function InterviewSimulation({ onBack }: { onBack: () => void }) {
  const [step, setStep]       = useState<SimStep>('mode');
  const [modes, setModes]     = useState<InterviewMode[]>([]);
  const [selectedMode, setSelectedMode] = useState<InterviewMode | null>(null);
  const [difficulty, setDifficulty]     = useState('intermediate');
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);

  // Interview state
  const [history, setHistory]               = useState<InterviewTurn[]>([]);
  const [coveredTopics, setCoveredTopics]   = useState<string[]>([]);
  const [status, setStatus]                 = useState<SimStatus>('idle');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [transcript, setTranscript]         = useState('');
  const [sessionTimer, setSessionTimer]     = useState(0);
  const [questionIndex, setQuestionIndex]   = useState(0);
  const [ttsAvailable, setTtsAvailable]     = useState<boolean | null>(null);

  // UI state
  const [isMicOn, setIsMicOn]           = useState(true);
  const [isVideoOn, setIsVideoOn]       = useState(true);
  const [showInsights, setShowInsights] = useState(false);
  const [isMuted, setIsMuted]           = useState(false);
  const [usePremiumModel, setUsePremiumModel] = useState(false);

  // Avatar state
  const [avatarState, setAvatarState] = useState<AvatarState>({
    emotion: 'NEUTRAL', pose: 'PROFESSIONAL_DEFAULT', gaze: 'DIRECT', micro_expression: 'NONE',
  });
  const [vocalParams, setVocalParams] = useState<VocalParams>({});
  const [mouthOpenness, setMouthOpenness] = useState(0);

  // Candidate metrics
  const [candidateMetrics, setCandidateMetrics] = useState<CandidateMetrics | null>(null);

  // Resume / pitch context for generative dialogue
  const [resumeContext]  = useState(() => localStorage.getItem('resume_analysis_context') || '');
  const [pitchContext]   = useState(() => localStorage.getItem('transcript_hint') || '');

  // Anti-Cheat state
  const [cheatWarning, setCheatWarning] = useState<string | null>(null);

  // Refs
  const physicsEngineRef  = useRef(new AvatarPhysicsEngine());
  const behaviorEngineRef = useRef(new HRBehaviorEngine(physicsEngineRef.current));
  const behaviorTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionTimerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const mediaRecorderRef  = useRef<MediaRecorder | null>(null);
  const chunksRef         = useRef<BlobPart[]>([]);
  const userVideoRef      = useRef<HTMLVideoElement>(null);
  const userVideoStreamRef = useRef<MediaStream | null>(null);
  const recognitionRef    = useRef<any>(null);
  const audioCtxRef       = useRef<AudioContext | null>(null);
  const analyserRef       = useRef<AnalyserNode | null>(null);
  const lipSyncRafRef     = useRef<number>(0);
  const currentSourceRef  = useRef<AudioBufferSourceNode | null>(null);
  const preFetchedAudioRef = useRef<AudioBuffer | null>(null);

  // VAD Refs
  const vadAudioCtxRef    = useRef<AudioContext | null>(null);
  const vadAnalyserRef    = useRef<AnalyserNode | null>(null);
  const vadStreamRef      = useRef<MediaStream | null>(null);
  const vadRafRef         = useRef<number>(0);
  const lastSpeechTimeRef = useRef<number>(0);
  const isSpeakingRef     = useRef<boolean>(false);

  // ── Cleanup ─────────────────────────────────────────────────────────────────
  const stopVideoStream = () => {
    userVideoStreamRef.current?.getTracks().forEach(t => t.stop());
    userVideoStreamRef.current = null;
    if (userVideoRef.current) userVideoRef.current.srcObject = null;
  };

  const stopAudioCapture = () => {
    try { recognitionRef.current?.stop(); } catch {}
    recognitionRef.current = null;
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== 'inactive') {
      rec.ondataavailable = null; rec.onstop = null;
      try { rec.stop(); } catch {}
    }
    rec?.stream?.getTracks().forEach(t => t.stop());
    mediaRecorderRef.current = null;

    cancelAnimationFrame(vadRafRef.current);
    vadStreamRef.current?.getTracks().forEach(t => t.stop());
    vadStreamRef.current = null;
    vadAudioCtxRef.current?.close();
    vadAudioCtxRef.current = null;
    vadAnalyserRef.current = null;
  };

  const stopLipSync = () => {
    cancelAnimationFrame(lipSyncRafRef.current);
    analyserRef.current = null;
    setMouthOpenness(0);
    physicsEngineRef.current.setMouthTarget(0);
  };

  const leaveSimulation = useCallback(() => {
    stopAudioCapture();
    window.speechSynthesis.cancel();
    currentSourceRef.current?.stop();
    stopVideoStream();
    stopLipSync();
    if (behaviorTimerRef.current) clearInterval(behaviorTimerRef.current);
    if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
    onBack();
  }, [onBack]);

  // ── Initial setup ────────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(apiUrl('/interview/modes'))
      .then(r => r.json())
      .then(d => setModes(d.modes?.length ? d.modes : DEFAULT_MODES))
      .catch(() => setModes(DEFAULT_MODES));

    // Check TTS availability
    fetch(apiUrl('/interview/tts/status'))
      .then(r => r.json())
      .then(d => setTtsAvailable(d.available))
      .catch(() => setTtsAvailable(false));

    // Behavior engine loop
    behaviorTimerRef.current = setInterval(() => {
      behaviorEngineRef.current.update(100);
      const cfg = behaviorEngineRef.current.config;
      setAvatarState(prev => ({
        ...prev,
        emotion: cfg.emotion as any,
        pose: cfg.pose as any,
        gaze: cfg.gaze as any,
        micro_expression: cfg.microExpression as any,
      }));
    }, 100);

    return () => {
      stopAudioCapture();
      window.speechSynthesis.cancel();
      stopVideoStream();
      stopLipSync();
      if (behaviorTimerRef.current) clearInterval(behaviorTimerRef.current);
      if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
    };
  }, []);

  // ── Webcam ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    if (!isVideoOn) { stopVideoStream(); return; }
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
      .then(stream => {
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        userVideoStreamRef.current = stream;
        if (userVideoRef.current) userVideoRef.current.srcObject = stream;
      })
      .catch(() => setIsVideoOn(false));
    return () => { cancelled = true; stopVideoStream(); };
  }, [isVideoOn]);

  // ── Session timer ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (step === 'interview') {
      sessionTimerRef.current = setInterval(() => setSessionTimer(t => t + 1), 1000);
    } else {
      if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
    }
    return () => { if (sessionTimerRef.current) clearInterval(sessionTimerRef.current); };
  }, [step]);

  // ── Lip sync via AudioAnalyser ───────────────────────────────────────────────
  const startLipSync = (analyser: AnalyserNode) => {
    analyserRef.current = analyser;
    const data = new Uint8Array(analyser.frequencyBinCount);
    let smoothed = 0;

    const loop = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getByteFrequencyData(data);
      // Focus on speech frequencies (100–4000 Hz)
      const speechBins = data.slice(2, 40);
      const avg = speechBins.reduce((s, v) => s + v, 0) / speechBins.length;
      const normalized = avg / 255;
      // Smooth with weighted average
      smoothed = smoothed * 0.7 + normalized * 0.3;
      setMouthOpenness(smoothed);
      physicsEngineRef.current.setMouthTarget(smoothed);
      lipSyncRafRef.current = requestAnimationFrame(loop);
    };
    loop();
  };

  // ── Reusable Play Audio Buffer & Lip Sync function ───────────────────────────
  const playAudioBuffer = useCallback((decoded: AudioBuffer, text: string) => {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContext();
    }
    const ctx = audioCtxRef.current;
    
    // Build audio processing chain
    const source    = ctx.createBufferSource();
    const analyser  = ctx.createAnalyser();
    const compressor = ctx.createDynamicsCompressor();

    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.8;
    compressor.knee.value = 12;
    compressor.ratio.value = 3.5;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.15;

    // Optional: subtle EQ for warm "room" feel
    const filter = ctx.createBiquadFilter();
    filter.type = 'highpass';
    filter.frequency.value = 80; // cut very low rumble
    filter.Q.value = 0.7;

    source.buffer = decoded;
    source.connect(analyser);
    analyser.connect(filter);
    filter.connect(compressor);
    compressor.connect(ctx.destination);

    currentSourceRef.current = source;
    startLipSync(analyser);

    setStatus('ai_speaking');
    source.start();
    source.onended = () => {
      stopLipSync();
      setAvatarState(prev => ({ ...prev, pose: 'PROFESSIONAL_DEFAULT', gaze: 'DIRECT' }));
      startListeningPhase();
    };
  }, [startListeningPhase, stopLipSync]);

  // ── Kokoro TTS (with Web Speech fallback) ────────────────────────────────────
  const speak = useCallback(async (text: string, params?: VocalParams) => {
    const vp = params || vocalParams;
    const fullText = vp.filler_prefix ? `${vp.filler_prefix} ${text}` : text;
    const pauseMs = vp.pause_before_ms || 700;

    setStatus('tts_loading');
    setCurrentQuestion(text);

    // Avatar: thinking pose while loading
    setAvatarState(prev => ({ ...prev, pose: 'THINKER_POSE', gaze: 'THINKING' }));

    await new Promise(r => setTimeout(r, pauseMs));

    if (isMuted) { setStatus('user_listening'); return; }

    // ── Try Kokoro TTS first ──
    if (ttsAvailable) {
      try {
        const fd = new FormData();
        fd.append('text', fullText);
        fd.append('voice_profile', selectedMode?.voiceProfile || 'friendly_hr');
        fd.append('speed', String(vp.speed_multiplier || 1.0));
        fd.append('pitch', String(vp.pitch_multiplier || 1.0));
        fd.append('pause_before_ms', String(vp.pause_before_ms || 0));

        const res = await fetch(apiUrl('/interview/tts'), {
          method: 'POST',
          headers: apiHeaders(),
          body: fd,
        });

        if (res.ok) {
          const arrayBuffer = await res.arrayBuffer();
          if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
            audioCtxRef.current = new AudioContext();
          }
          const ctx = audioCtxRef.current;
          const decoded = await ctx.decodeAudioData(arrayBuffer);
          playAudioBuffer(decoded, text);
          return;
        }
      } catch (e) {
        console.warn('Kokoro TTS failed, falling back to Web Speech:', e);
      }
    }

    // ── Web Speech API fallback ──
    setStatus('ai_speaking');
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(fullText);
    const voices = window.speechSynthesis.getVoices();
    const premiumVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Premium'))
      || voices.find(v => v.lang.startsWith('en'))
      || voices[0];
    if (premiumVoice) utterance.voice = premiumVoice;
    utterance.rate = Math.max(0.7, Math.min(1.3, 0.95 * (vp.speed_multiplier || 1.0)));
    utterance.pitch = 1.0 * (vp.pitch_multiplier || 1.0);
    utterance.onend = () => {
      setAvatarState(prev => ({ ...prev, pose: 'PROFESSIONAL_DEFAULT', gaze: 'DIRECT' }));
      startListeningPhase();
    };
    window.speechSynthesis.speak(utterance);
  }, [vocalParams, ttsAvailable, selectedMode, isMuted, playAudioBuffer, startListeningPhase]);

  // ── Start interview ──────────────────────────────────────────────────────────
  const startInterview = async (mode: InterviewMode) => {
    window.speechSynthesis.cancel();
    const unlock = new SpeechSynthesisUtterance(''); unlock.volume = 0;
    window.speechSynthesis.speak(unlock);

    setSelectedMode(mode);
    setStep('loading');
    setSimulationResult(null);
    setTranscript('');
    setQuestionIndex(0);
    setCoveredTopics([]);

    const greeting = `Hi! I'm going to be your interviewer for the ${mode.name} round today. Why don't you start by briefly telling me about yourself and your background?`;

    // Start pre-fetching greeting audio immediately to avoid transition latency
    preFetchedAudioRef.current = null;
    const fetchPromise = (async () => {
      try {
        const fd = new FormData();
        fd.append('text', greeting);
        fd.append('voice_profile', mode.voiceProfile || 'friendly_hr');
        fd.append('speed', '1.0');
        fd.append('pitch', '1.0');
        fd.append('pause_before_ms', '0');

        const res = await fetch(apiUrl('/interview/tts'), {
          method: 'POST',
          headers: apiHeaders(),
          body: fd,
        });
        if (res.ok) {
          const arrayBuffer = await res.arrayBuffer();
          if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
            audioCtxRef.current = new AudioContext();
          }
          const decoded = await audioCtxRef.current.decodeAudioData(arrayBuffer);
          return decoded;
        }
      } catch (err) {
        console.warn('Pre-fetch greeting audio failed:', err);
      }
      return null;
    })();

    setTimeout(async () => {
      setStep('interview');
      setHistory([{ role: 'assistant', content: greeting }]);
      setAvatarState({ emotion: 'NEUTRAL', pose: 'PROFESSIONAL_DEFAULT', gaze: 'DIRECT', micro_expression: 'SUBTLE_SMILE' });
      
      const decodedBuffer = await fetchPromise;
      if (decodedBuffer) {
        playAudioBuffer(decodedBuffer, greeting);
      } else {
        speak(greeting);
      }
    }, 1800);
  };

  // ── Recording & VAD ────────────────────────────────────────────────────────
  const startListeningPhase = async () => {
    setStatus('user_listening');
    if (!isMicOn) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : undefined);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      setTranscript('');

      const SpeechRec = window.SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRec) {
        const rec = new SpeechRec();
        rec.continuous = true; rec.interimResults = true; rec.lang = 'en-US';
        rec.onresult = (e: any) => {
          let spoken = '';
          for (let i = 0; i < e.results.length; i++) spoken += e.results[i][0].transcript;
          setTranscript(spoken.trim());
        };
        rec.onerror = () => { recognitionRef.current = null; };
        recognitionRef.current = rec;
        rec.start();
      }

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        processUserResponse(blob);
        stream.getTracks().forEach(t => t.stop());
      };
      mr.start();

      // Start VAD Loop
      const audioCtx = new AudioContext();
      vadAudioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);
      vadAnalyserRef.current = analyser;
      vadStreamRef.current = stream;
      isSpeakingRef.current = false;
      lastSpeechTimeRef.current = Date.now(); // reset timer

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      const checkAudioLevel = () => {
        if (!vadAnalyserRef.current) return;
        analyser.getByteFrequencyData(dataArray);
        
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        const average = sum / dataArray.length;
        
        const THRESHOLD = 20;
        const SILENCE_TIMEOUT = 4500; // 4.5s of silence triggers end
        
        const now = Date.now();
        if (average > THRESHOLD) {
          lastSpeechTimeRef.current = now;
          if (!isSpeakingRef.current) {
             isSpeakingRef.current = true;
             setStatus('user_speaking');
          }
        } else {
           if (isSpeakingRef.current && (now - lastSpeechTimeRef.current > SILENCE_TIMEOUT)) {
             isSpeakingRef.current = false;
             stopRecording();
             return;
           }
        }
        vadRafRef.current = requestAnimationFrame(checkAudioLevel);
      };
      
      checkAudioLevel();

    } catch (err) {
      console.warn("VAD failed to start", err);
    }
  };

  const stopRecording = () => {
    cancelAnimationFrame(vadRafRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { recognitionRef.current?.stop(); } catch {}
      recognitionRef.current = null;
      mediaRecorderRef.current.stop();
      setStatus('processing');
    }
  };

  // ── Process response ─────────────────────────────────────────────────────────
  const processUserResponse = async (blob: Blob) => {
    setStatus('processing');
    setAvatarState(prev => ({ ...prev, pose: 'THINKER_POSE', emotion: 'THOUGHTFUL', gaze: 'THINKING' }));

    try {
      const fd = new FormData();
      fd.append('file', blob, 'response.webm');
      fd.append('history', JSON.stringify(history));
      fd.append('user_response_hint', transcript);
      fd.append('difficulty', difficulty);
      fd.append('mode', selectedMode?.id || 'hr');
      fd.append('resume_text', resumeContext);
      fd.append('pitch_text', pitchContext);
      fd.append('covered_topics', JSON.stringify(coveredTopics));
      fd.append('use_premium_model', String(usePremiumModel));
      if (candidateMetrics) {
        fd.append('candidate_metrics', JSON.stringify(candidateMetrics));
      }

      const res = await fetch(apiUrl('/interview/simulate/respond'), {
        method: 'POST', headers: apiHeaders(), body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const turn = await res.json() as SimulationTurnResponse;
      const spokenText = turn.transcript || transcript || 'I need a moment to formulate my answer.';
      const userTurn: InterviewTurn = { role: 'user', content: spokenText };
      const updatedHistory: InterviewTurn[] = [...history, userTurn];

      // Update avatar state from backend
      if (turn.avatar_state) {
        setAvatarState(turn.avatar_state);
        behaviorEngineRef.current.reactToAvatarState(
          turn.avatar_state.emotion,
          turn.avatar_state.pose,
          turn.avatar_state.gaze,
        );
      }
      if (turn.vocal_params) setVocalParams(turn.vocal_params);

      // Track covered topics for generative diversity
      if (turn.topic) setCoveredTopics(prev => [...new Set([...prev, turn.topic!])]);

      if (turn.should_end) {
        const closing = turn.final_summary || turn.feedback || 'That wraps up our session. I have enough to evaluate your candidacy.';
        setHistory([...updatedHistory, { role: 'assistant', content: closing }]);
        setSimulationResult(turn);
        setAvatarState({ emotion: 'CONCLUDING', pose: 'OPEN_PALMS', gaze: 'DIRECT', micro_expression: 'SUBTLE_SMILE' });
        setStatus('idle');
        setStep('result');
      } else {
        const nextQ = turn.next_question || 'Can you give me a concrete example that demonstrates that?';
        const hrResponse = turn.hr_response ? `${turn.hr_response} ${nextQ}` : nextQ;
        const newHistory: InterviewTurn[] = [...updatedHistory, { role: 'assistant', content: nextQ, topic: turn.topic }];
        setHistory(newHistory);
        setQuestionIndex(i => i + 1);
        setTranscript('');
        speak(hrResponse, turn.vocal_params);

        // Candidate signal → behavior engine
        const cue = turn.behavior_cue;
        if (cue === 'impressed') behaviorEngineRef.current.reactToSignal('STRONG_ANSWER');
        else if (cue === 'skeptical') behaviorEngineRef.current.reactToSignal('VAGUE_ANSWER');
        else if (cue === 'evaluating') behaviorEngineRef.current.reactToSignal('WEAK_ANSWER');
      }
    } catch (err) {
      console.error('Simulation error:', err);
      // Intelligent fallback
      const spokenText = transcript || 'I need a moment to answer that.';
      const updatedHistory = [...history, { role: 'user', content: spokenText }];
      const userTurns = updatedHistory.filter(t => t.role === 'user').length;

      if (userTurns >= 4) {
        const localResult = buildLocalResult(updatedHistory);
        setHistory([...updatedHistory, { role: 'assistant', content: localResult.final_summary || 'Session complete.' }]);
        setSimulationResult(localResult);
        setStep('result');
        return;
      }

      const fallbacks = [
        "That's interesting — can you walk me through a specific example with a measurable outcome?",
        "I want to understand your thought process better. What was the key decision you made there?",
        "What tradeoffs did you consider in that situation?",
        "How did you measure the success of that work?",
      ];
      const nextQ = fallbacks[userTurns % fallbacks.length];
      setHistory([...updatedHistory, { role: 'assistant', content: nextQ }]);
      setTranscript('');
      speak(nextQ);
    }
  };

  const buildLocalResult = (hist: InterviewTurn[]): SimulationResult => {
    const answers = hist.filter(t => t.role === 'user').map(t => t.content).join(' ');
    const words = answers.trim().split(/\s+/).filter(Boolean);
    const evidenceHits = (answers.match(/\b(project|built|led|improved|designed|launched|team|metric|python|react|data|model)\b/gi) || []).length;
    const score = Math.max(4, Math.min(9.5, 5.5 + Math.min(2, words.length / 80) + Math.min(1.8, evidenceHits * 0.22)));
    return {
      score: Number(score.toFixed(1)),
      final_summary: 'Session complete. Focus on making every answer specific, evidence-backed, and tied to measurable results.',
      strengths: [
        evidenceHits > 0 ? 'You referenced concrete experience with action verbs and evidence.' : 'You stayed engaged throughout the session.',
        'You completed a full multi-turn interview flow.',
      ],
      improvements: [
        'Close every answer with a measurable outcome, not just the task performed.',
        'Use the STAR structure for behavioral questions.',
      ],
    };
  };

  // ── Candidate signal handler ─────────────────────────────────────────────────
  const handleCandidateSignal = useCallback((signal: CandidateSignal) => {
    behaviorEngineRef.current.reactToSignal(signal);
    if (signal === 'READING_DETECTED' || signal === 'EYE_CONTACT_LOST') {
      setCheatWarning('⚠️ Please maintain eye contact and do not read from a script.');
      setTimeout(() => setCheatWarning(null), 5000);
    }
  }, []);

  // ── Timer format ─────────────────────────────────────────────────────────────
  const formatTime = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  // ── Total questions estimate ─────────────────────────────────────────────────
  const totalQ = selectedMode?.question_count || 5;

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="fixed inset-0 z-[100] bg-[#050508] text-white flex flex-col font-sans overflow-hidden">

      {/* ═══ MODE SELECTION SCREEN ═══ */}
      {step === 'mode' && (
        <FadeIn className="flex-1 flex items-center justify-center p-8 bg-[#050508]">
          <div className="max-w-5xl w-full">
            {/* Header */}
            <div className="mb-10">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-indigo-600/20 rounded-xl border border-indigo-500/30">
                  <Zap className="w-6 h-6 text-indigo-400" />
                </div>
                <div>
                  <h1 className="text-4xl font-black tracking-tighter">AI Interview Room</h1>
                  <p className="text-white/40 text-sm mt-0.5">Select your interview mode and difficulty level</p>
                </div>
              </div>

              {/* Difficulty Pills */}
              <div className="flex flex-wrap gap-2 mt-6">
                {Object.entries(DIFFICULTY_CONFIG).map(([key, cfg]) => (
                  <button
                    key={key}
                    onClick={() => setDifficulty(key)}
                    className={`px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-widest border transition-all ${
                      difficulty === key ? cfg.color : 'bg-white/5 border-white/10 text-white/40 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    {cfg.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Mode Cards */}
            <div className="grid md:grid-cols-2 gap-4">
              {(modes.length ? modes : DEFAULT_MODES).map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => startInterview(mode)}
                  className="group relative bg-white/[0.04] border border-white/10 p-7 rounded-2xl text-left
                    hover:bg-white/[0.08] hover:border-indigo-500/40 transition-all duration-300
                    hover:shadow-[0_0_40px_rgba(99,102,241,0.12)]"
                >
                  <div className="text-3xl mb-3">{mode.icon || '💼'}</div>
                  <h3 className="text-xl font-bold mb-1.5 group-hover:text-indigo-300 transition-colors">{mode.name}</h3>
                  <p className="text-white/40 text-sm mb-5 leading-relaxed">{mode.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="px-3 py-1 bg-white/5 rounded-full text-[10px] font-bold uppercase tracking-widest text-white/50">
                      ~{mode.question_count} questions
                    </span>
                    <span className="flex items-center gap-1 text-indigo-400 text-xs font-bold group-hover:gap-2 transition-all">
                      Start <ChevronRight className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </button>
              ))}
            </div>

            {/* TTS Status */}
            <div className="flex items-center gap-2 mt-6">
              {ttsAvailable === true && (
                <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                  <span className="text-[10px] text-emerald-400 font-bold">Kokoro TTS Active — Human Voice Ready</span>
                </div>
              )}
              {ttsAvailable === false && (
                <div className="flex items-center gap-1.5 px-3 py-1 bg-amber-500/10 border border-amber-500/20 rounded-full">
                  <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                  <span className="text-[10px] text-amber-400 font-bold">Using browser TTS — Run /interview/tts/setup for human voice</span>
                </div>
              )}
            </div>

            <button onClick={leaveSimulation} className="mt-8 text-white/30 hover:text-white/60 transition-colors text-sm font-bold">
              ← Back to Dashboard
            </button>
          </div>
        </FadeIn>
      )}

      {/* ═══ LOADING SCREEN ═══ */}
      {step === 'loading' && (
        <div className="flex-1 flex flex-col items-center justify-center bg-[#050508]">
          <div className="relative w-20 h-20 mb-8">
            <div className="absolute inset-0 rounded-full border-4 border-indigo-500/20" />
            <div className="absolute inset-0 rounded-full border-4 border-t-indigo-500 animate-spin" />
          </div>
          <h2 className="text-2xl font-bold animate-pulse">Initializing Interview Room...</h2>
          <p className="text-white/30 mt-2 text-sm">Loading {selectedMode?.name} session</p>
        </div>
      )}

      {/* ═══ RESULTS SCREEN ═══ */}
      {step === 'result' && simulationResult && (
        <FadeIn className="flex-1 flex items-center justify-center p-8 bg-[#050508]/98 backdrop-blur-md overflow-auto">
          <div className="max-w-5xl w-full grid lg:grid-cols-[1fr_1.4fr] gap-5">
            {/* Score Panel */}
            <div className="bg-white/[0.04] border border-white/10 rounded-2xl p-8 flex flex-col">
              <p className="text-[10px] font-black uppercase tracking-[0.25em] text-indigo-300 mb-2">Overall Score</p>
              <div className="flex items-end gap-2 mb-1">
                <span className="text-8xl font-black tracking-tighter leading-none">
                  {Math.round((simulationResult.score || 6.5) * 10)}
                </span>
                <span className="text-white/30 font-bold mb-3 text-lg">/100</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden mb-6">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-1000"
                  style={{ width: `${Math.round((simulationResult.score || 6.5) * 10)}%` }}
                />
              </div>
              <p className="text-white/55 leading-relaxed text-sm flex-1">
                {simulationResult.final_summary || simulationResult.feedback}
              </p>
              <div className="flex gap-3 mt-8">
                <button
                  onClick={() => { setStep('mode'); setHistory([]); setCurrentQuestion(''); setSimulationResult(null); setQuestionIndex(0); setCoveredTopics([]); }}
                  className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-bold text-sm transition-colors"
                >
                  Practice Again
                </button>
                <button
                  onClick={leaveSimulation}
                  className="flex-1 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl font-bold text-sm transition-colors"
                >
                  Dashboard
                </button>
              </div>
            </div>

            {/* Transcript + Feedback Panel */}
            <div className="bg-white/[0.04] border border-white/10 rounded-2xl p-8 max-h-[78vh] overflow-y-auto custom-scrollbar">
              <div className="grid sm:grid-cols-2 gap-4 mb-8">
                <div>
                  <h3 className="text-xs font-black uppercase tracking-widest text-emerald-400 mb-3">Strengths</h3>
                  <div className="space-y-2">
                    {(simulationResult.strengths || []).map((s, i) => (
                      <div key={i} className="text-sm text-white/65 bg-emerald-500/5 border border-emerald-500/15 rounded-xl p-3 leading-relaxed">{s}</div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-xs font-black uppercase tracking-widest text-amber-400 mb-3">Next Drills</h3>
                  <div className="space-y-2">
                    {(simulationResult.improvements || []).map((s, i) => (
                      <div key={i} className="text-sm text-white/65 bg-amber-500/5 border border-amber-500/15 rounded-xl p-3 leading-relaxed">{s}</div>
                    ))}
                  </div>
                </div>
              </div>
              <h3 className="text-xs font-black uppercase tracking-widest text-white/30 mb-3">Conversation Transcript</h3>
              <div className="space-y-2.5">
                {history.map((turn, i) => (
                  <div key={i} className={`p-3.5 rounded-xl text-sm border ${turn.role === 'assistant' ? 'bg-white/[0.04] border-white/8' : 'bg-indigo-500/8 border-indigo-500/20'}`}>
                    <p className="font-black uppercase text-[9px] tracking-widest mb-1.5 opacity-40">{turn.role === 'assistant' ? 'Interviewer' : 'You'}</p>
                    <p className="text-white/70 leading-relaxed">{turn.content}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>
      )}

      {/* ═══ MAIN INTERVIEW ROOM ═══ */}
      {step === 'interview' && (
        <>
          {/* ── Top Bar ── */}
          <div className="h-14 flex items-center justify-between px-6 bg-black/40 backdrop-blur-xl border-b border-white/[0.06] z-30 shrink-0">
            <div className="flex items-center gap-3">
              <div className="p-1.5 bg-indigo-600/20 rounded-lg border border-indigo-500/30">
                <Zap className="w-4 h-4 text-indigo-400" />
              </div>
              <div>
                <p className="text-sm font-bold leading-tight">{selectedMode?.name || 'Interview'}</p>
                <div className="flex items-center gap-1.5">
                  <Shield className="w-2.5 h-2.5 text-emerald-400" />
                  <p className="text-[9px] text-white/35 uppercase font-black tracking-widest">{difficulty} mode</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Session Timer */}
              <div className="flex items-center gap-1.5 text-white/40">
                <Clock className="w-3.5 h-3.5" />
                <span className="text-xs font-mono font-bold">{formatTime(sessionTimer)}</span>
              </div>

              {/* Question Progress */}
              <div className="hidden sm:flex items-center gap-1.5">
                {Array.from({ length: totalQ }).map((_, i) => (
                  <div key={i} className={`w-1.5 h-1.5 rounded-full transition-all duration-500 ${i < questionIndex ? 'bg-indigo-500' : i === questionIndex ? 'bg-white/60 scale-125' : 'bg-white/15'}`} />
                ))}
              </div>

              <span className="text-xs text-white/35 font-medium hidden md:block">
                Q {Math.min(questionIndex + 1, totalQ)} / {totalQ}
              </span>
              
              {/* Premium Models Toggle */}
              <div className="flex items-center gap-2 bg-indigo-500/10 px-3 py-1.5 rounded-full border border-indigo-500/20 ml-2">
                <span className="text-xs font-semibold text-indigo-300">Premium (Gemini)</span>
                <button
                  onClick={() => setUsePremiumModel(!usePremiumModel)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${usePremiumModel ? 'bg-indigo-500' : 'bg-white/20'}`}
                >
                  <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${usePremiumModel ? 'translate-x-5' : 'translate-x-1'}`} />
                </button>
              </div>
            </div>
          </div>

          {/* ── Main Interview Area ── */}
          <div className="flex-1 flex overflow-hidden">

            {/* ── LEFT: Interviewer Panel ── */}
            <div className="flex-1 relative flex flex-col p-5 pb-0 min-w-0">

              {/* HR Avatar — fills the panel */}
              <div className="flex-1 relative rounded-2xl overflow-hidden bg-[#1a1410] z-0">
                <Canvas
                  camera={{ position: [0, 0.45, 2.95], fov: 44, near: 0.1, far: 50 }}
                  shadows
                  gl={{ antialias: true, toneMapping: 3 /* ACESFilmic */ }}
                  style={{ background: 'transparent' }}
                >
                  <Suspense fallback={null}>
                    <Avatar3D
                      isSpeaking={status === 'ai_speaking'}
                      isThinking={status === 'processing' || status === 'tts_loading'}
                      mouthOpenness={mouthOpenness}
                    />
                  </Suspense>
                </Canvas>

                {/* Current question subtitle */}
                {status === 'ai_speaking' && currentQuestion && (
                  <div className="absolute bottom-5 left-0 right-0 px-8 z-20 pointer-events-none">
                    <div className="max-w-xl mx-auto bg-black/50 backdrop-blur-2xl border border-white/10 px-6 py-4 rounded-2xl">
                      <p className="text-[13px] font-medium text-white/85 leading-relaxed italic text-center">
                        {currentQuestion}
                      </p>
                    </div>
                  </div>
                )}

                {/* Live transcript while user speaks */}
                {status === 'user_speaking' && transcript && (
                  <div className="absolute bottom-5 left-0 right-0 px-8 z-20 pointer-events-none">
                    <div className="max-w-xl mx-auto bg-indigo-900/40 backdrop-blur-2xl border border-indigo-500/25 px-6 py-4 rounded-2xl">
                      <p className="text-[13px] text-indigo-200/85 leading-relaxed text-center">
                        {transcript}
                      </p>
                    </div>
                  </div>
                )}

                {/* Listening indicator */}
                {status === 'user_listening' && !currentQuestion && (
                  <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
                    <div className="px-5 py-2.5 bg-black/50 backdrop-blur-xl border border-white/10 rounded-full">
                      <p className="text-[11px] text-white/50 font-bold uppercase tracking-widest">Your Turn</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── RIGHT: Candidate Panel ── */}
            <div className="w-80 shrink-0 flex flex-col p-5 pl-0 gap-4">

              {/* Webcam feed */}
              <div className="relative bg-[#0d0d14] rounded-2xl overflow-hidden border border-white/[0.07] aspect-video">
                <video
                  ref={userVideoRef}
                  autoPlay muted playsInline
                  className="w-full h-full object-cover"
                />
                {!isVideoOn && (
                  <div className="absolute inset-0 flex items-center justify-center bg-[#0d0d14]">
                    <VideoOff className="w-8 h-8 text-white/15" />
                  </div>
                )}
                <div className="absolute bottom-2 left-3 text-[8px] font-black uppercase tracking-widest text-white/40">You</div>

                {/* Speaking indicator on webcam */}
                {status === 'user_speaking' && (
                  <div className="absolute inset-0 rounded-2xl border-2 border-indigo-500/60 animate-speaking-pulse pointer-events-none" />
                )}
              </div>

              {/* Candidate Observation Metrics */}
              <div className="bg-white/[0.04] border border-white/[0.07] rounded-2xl p-4 flex-1 flex flex-col">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                  <span className="text-[10px] font-black uppercase tracking-widest text-white/40">Live Analysis</span>
                </div>

                <CandidateObserver
                  videoRef={userVideoRef}
                  isActive={step === 'interview' && isVideoOn}
                  onMetricsUpdate={setCandidateMetrics}
                  onSignal={handleCandidateSignal}
                />
              </div>

              {/* HR Mood Indicator */}
              <div className="bg-white/[0.04] border border-white/[0.07] rounded-2xl p-4">
                <span className="text-[10px] font-black uppercase tracking-widest text-white/30 block mb-3">HR Status</span>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    status === 'ai_speaking' ? 'bg-blue-400 animate-pulse' :
                    status === 'processing' ? 'bg-amber-400 animate-pulse' :
                    status === 'user_listening' ? 'bg-emerald-400' :
                    status === 'user_speaking' ? 'bg-indigo-400' : 'bg-white/30'
                  }`} />
                  <span className="text-sm font-medium text-white/60">
                    {status === 'ai_speaking' ? 'Speaking...' :
                     status === 'tts_loading' ? 'Thinking...' :
                     status === 'processing' ? 'Evaluating...' :
                     status === 'user_listening' ? 'Listening' :
                     status === 'user_speaking' ? 'Recording you' : 'Ready'}
                  </span>
                </div>
                {avatarState.emotion !== 'NEUTRAL' && (
                  <div className="mt-2 text-[10px] text-white/25 font-medium">
                    Mood: {avatarState.emotion.charAt(0) + avatarState.emotion.slice(1).toLowerCase()}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Bottom Control Bar ── */}
          <div className="h-20 flex items-center justify-center gap-5 px-8 bg-black/30 backdrop-blur-2xl border-t border-white/[0.06] shrink-0">

            {/* Mic toggle */}
            <button
              id="sim-mic-toggle"
              onClick={() => setIsMicOn(!isMicOn)}
              className={`p-3.5 rounded-full border transition-all ${isMicOn ? 'bg-white/[0.06] border-white/15 text-white hover:bg-white/10' : 'bg-red-500 border-red-400 text-white'}`}
            >
              {isMicOn ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
            </button>

            {/* Camera toggle */}
            <button
              id="sim-cam-toggle"
              onClick={() => setIsVideoOn(!isVideoOn)}
              className={`p-3.5 rounded-full border transition-all ${isVideoOn ? 'bg-white/[0.06] border-white/15 text-white hover:bg-white/10' : 'bg-red-500 border-red-400 text-white'}`}
            >
              {isVideoOn ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
            </button>

            {/* Mute TTS */}
            <button
              onClick={() => setIsMuted(!isMuted)}
              className={`p-3.5 rounded-full border transition-all ${!isMuted ? 'bg-white/[0.06] border-white/15 text-white hover:bg-white/10' : 'bg-amber-500/20 border-amber-500/40 text-amber-400'}`}
            >
              {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>

            <div className="h-8 w-px bg-white/10" />

            {/* Main action area: Replaced manual buttons with VAD indicators */}
            {status === 'user_listening' && (
              <div className="px-8 py-3.5 rounded-full flex items-center gap-3 font-bold text-sm transition-all bg-white/10 text-white/50">
                <span className="w-2 h-2 bg-white/30 rounded-full" />
                Listening for you to speak...
              </div>
            )}

            {status === 'user_speaking' && (
              <div className="px-8 py-3.5 bg-indigo-600 rounded-full flex items-center gap-3 font-bold text-sm transition-all shadow-[0_0_30px_rgba(99,102,241,0.35)] animate-respond-pulse">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                Recording your answer...
              </div>
            )}

            {status === 'processing' && (
              <div className="px-8 py-3.5 bg-white/5 rounded-full flex items-center gap-3 font-bold text-sm text-white/40">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white/80 rounded-full animate-spin" />
                Processing answer...
              </div>
            )}
            
            {status === 'tts_loading' && (
              <div className="px-8 py-3.5 bg-white/5 rounded-full flex items-center gap-3 font-bold text-sm text-white/40">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white/80 rounded-full animate-spin" />
                HR is thinking...
              </div>
            )}

            {status === 'ai_speaking' && (
              <div className="px-8 py-3.5 bg-blue-500/10 border border-blue-500/20 rounded-full flex items-center gap-3 font-bold text-sm text-blue-300">
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                Interviewer speaking
              </div>
            )}

            <div className="h-8 w-px bg-white/10" />

            {/* Insights toggle */}
            <button
              onClick={() => setShowInsights(!showInsights)}
              className={`p-3.5 rounded-full border transition-all ${showInsights ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-400' : 'bg-white/[0.06] border-white/15 text-white/60 hover:text-white'}`}
            >
              <MessageSquare className="w-5 h-5" />
            </button>

            {/* Leave */}
            <button
              id="sim-leave-btn"
              onClick={leaveSimulation}
              className="px-5 py-3.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded-full flex items-center gap-2 font-bold text-sm transition-all"
            >
              <PhoneOff className="w-4 h-4" />
              End
            </button>
          </div>

          {/* ── Sliding Insights Panel ── */}
          {showInsights && (
            <div className="absolute right-0 top-14 bottom-20 w-80 bg-[#0a0a10]/95 backdrop-blur-2xl border-l border-white/[0.07] flex flex-col p-5 z-40">
              <h3 className="font-bold text-sm flex items-center gap-2 mb-4">
                <MessageSquare className="w-4 h-4 text-indigo-400" />
                Session Transcript
              </h3>
              <div className="flex-1 overflow-y-auto space-y-3 pr-1 custom-scrollbar">
                {history.map((turn, i) => (
                  <div key={i} className={`p-3 rounded-xl text-sm border ${turn.role === 'assistant' ? 'bg-white/[0.04] border-white/8' : 'bg-indigo-500/8 border-indigo-500/20'}`}>
                    <p className="font-black uppercase text-[9px] tracking-widest mb-1 opacity-35">{turn.role === 'assistant' ? 'Interviewer' : 'You'}</p>
                    <p className="text-white/65 leading-relaxed text-xs">{turn.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Global CSS injected ── */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 3px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 10px; }

        @keyframes respond-pulse {
          0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
          70% { box-shadow: 0 0 0 18px rgba(99,102,241,0); }
          100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
        }
        .animate-respond-pulse { animation: respond-pulse 2s infinite; }

        @keyframes speaking-pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        .animate-speaking-pulse { animation: speaking-pulse 1.2s ease-in-out infinite; }

        @keyframes recording-ring {
          0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
          70% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
          100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
        }
        .recording-ring { animation: recording-ring 1.5s infinite; }
      `}</style>
    </div>
  );
}
