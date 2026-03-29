import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Mic, Square, Activity, Upload, FileAudio, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

type Mode = 'record' | 'upload';

export default function InterviewScreen({ onFinish }: { onFinish: (blob: Blob) => void }) {
  const [mode, setMode] = useState<Mode>('record');

  // ── RECORD state ──────────────────────────────────────────────────────────
  const [isRecording, setIsRecording] = useState(false);
  const [time, setTime] = useState(0);
  const [transcript, setTranscript] = useState('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recognitionRef = useRef<any>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // ── UPLOAD state ──────────────────────────────────────────────────────────
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Timer ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => setTime((t) => t + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  // Cleanup upload object URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  // ── Switch mode guard ─────────────────────────────────────────────────────
  const switchMode = (newMode: Mode) => {
    if (isRecording) return; // Don't switch mid-recording
    setMode(newMode);
    setTranscript('');
    setTime(0);
    if (uploadedFile) {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
      setUploadedFile(null);
    }
  };

  // ── RECORD handlers ───────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'interview.webm', { type: 'audio/webm' });
        onFinish(audioFile);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;

        recognitionRef.current.onresult = (event: any) => {
          let current = '';
          for (let i = 0; i < event.results.length; ++i) {
            current += event.results[i][0].transcript;
          }
          setTranscript(current);
        };
        recognitionRef.current.start();
      } else {
        setTranscript('Live transcription is not supported in this browser. Speak clearly — your audio is being recorded.');
      }

      setIsRecording(true);
    } catch (err) {
      console.error('Microphone access error:', err);
      onFinish(new Blob());
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      if (recognitionRef.current) recognitionRef.current.stop();
      setIsRecording(false);
    }
  };

  // ── UPLOAD handlers ───────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    console.log('[Upload] File selected:', { name: file.name, size: file.size, type: file.type });
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = URL.createObjectURL(file);
    setUploadedFile(file);
    // Reset input so the same file can be re-selected if needed
    e.target.value = '';
  };

  const handleUploadSubmit = () => {
    console.log('[Upload Submit] Button clicked. uploadedFile:', uploadedFile);
    if (!uploadedFile) {
      console.error('[Upload Submit] No file stored in state!');
      return;
    }
    console.log('[Upload Submit] Calling onFinish with File:', { name: uploadedFile.name, size: uploadedFile.size, type: uploadedFile.type });
    onFinish(uploadedFile);
  };

  const clearUpload = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setUploadedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Helpers ───────────────────────────────────────────────────────────────
  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl flex flex-col items-center gap-8"
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex justify-between items-center w-full px-2">
        <h2 className="text-2xl font-semibold tracking-tight text-slate-100">
          {mode === 'record' ? 'Live Session' : 'Upload Audio'}
        </h2>
        {mode === 'record' && (
          <div className="flex items-center gap-3">
            {isRecording && (
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
            )}
            <span className="text-2xl font-mono text-slate-400 font-medium">
              {formatTime(time)}
            </span>
          </div>
        )}
      </div>

      {/* ── Mode Switcher ────────────────────────────────────────────────── */}
      <div className="relative flex w-full max-w-xs bg-slate-800/60 rounded-xl p-1 border border-slate-700">
        {/* Animated slider background */}
        <motion.div
          className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg shadow-lg"
          animate={{ left: mode === 'record' ? '4px' : 'calc(50%)' }}
          transition={{ type: 'spring', stiffness: 400, damping: 35 }}
        />
        <button
          onClick={() => switchMode('record')}
          disabled={isRecording}
          className={`relative z-10 flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-lg text-sm font-medium transition-colors duration-200 ${
            mode === 'record' ? 'text-white' : 'text-slate-400 hover:text-slate-200'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          id="mode-record-tab"
        >
          <Mic className="w-4 h-4" />
          Record
        </button>
        <button
          onClick={() => switchMode('upload')}
          disabled={isRecording}
          className={`relative z-10 flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-lg text-sm font-medium transition-colors duration-200 ${
            mode === 'upload' ? 'text-white' : 'text-slate-400 hover:text-slate-200'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          id="mode-upload-tab"
        >
          <Upload className="w-4 h-4" />
          Upload
        </button>
      </div>

      {/* ── Content Panel ────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">

        {/* ── RECORD MODE ─────────────────────────────────────────────── */}
        {mode === 'record' && (
          <motion.div
            key="record-panel"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.2 }}
            className="w-full flex flex-col items-center gap-10"
          >
            <Card className="glass w-full h-72 py-6 px-8 flex flex-col relative overflow-hidden rounded-[2rem]">
              {!isRecording ? (
                <div className="text-slate-400 flex flex-col items-center justify-center h-full gap-4">
                  <div className="w-20 h-20 rounded-full bg-slate-800 flex items-center justify-center shadow-inner">
                    <Mic className="w-10 h-10 text-slate-500" />
                  </div>
                  <p className="text-lg tracking-wide">Press the microphone below to start</p>
                </div>
              ) : (
                <div className="flex flex-col h-full w-full">
                  <div className="flex items-center gap-4 mb-4 border-b border-slate-800 pb-3">
                    <Activity className="w-6 h-6 text-indigo-400 animate-pulse" />
                    <p className="text-indigo-200 text-sm font-medium tracking-widest uppercase animate-pulse">
                      Listening
                    </p>
                  </div>
                  <div className="flex-1 overflow-y-auto text-lg leading-relaxed text-slate-300 italic pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                    {transcript ? `"${transcript}"` : 'Speak now, transcription is active...'}
                  </div>
                  <div className="flex gap-1.5 items-end h-8 mt-4 w-full justify-center opacity-50">
                    {[...Array(24)].map((_, i) => (
                      <motion.div
                        key={i}
                        className="w-1 bg-indigo-500 rounded-full"
                        animate={{ height: ['20%', '100%', '20%'] }}
                        transition={{ repeat: Infinity, duration: 0.8 + Math.random(), ease: 'easeInOut' }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </Card>

            <div className="relative mt-4">
              <Button
                size="lg"
                id="record-toggle-btn"
                onClick={isRecording ? stopRecording : startRecording}
                className={`h-24 w-24 rounded-full shadow-2xl transition-all duration-300 ${
                  isRecording
                    ? 'bg-slate-800 hover:bg-slate-700 border border-slate-600'
                    : 'bg-gradient-to-br from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 border-0'
                } flex items-center justify-center`}
              >
                {isRecording ? (
                  <Square className="h-8 w-8 text-red-400" />
                ) : (
                  <Mic className="h-10 w-10 text-white" />
                )}
              </Button>
              {isRecording && (
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-indigo-500 pointer-events-none"
                  animate={{ scale: [1, 1.4], opacity: [1, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              )}
            </div>
          </motion.div>
        )}

        {/* ── UPLOAD MODE ─────────────────────────────────────────────── */}
        {mode === 'upload' && (
          <motion.div
            key="upload-panel"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="w-full flex flex-col items-center gap-6"
          >
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a,.flac"
              onChange={handleFileChange}
              className="hidden"
              id="audio-file-input"
            />

            {!uploadedFile ? (
              /* ── Drop zone ──────────────────────────────────────────── */
              <motion.div
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-64 rounded-[2rem] border-2 border-dashed border-slate-600 hover:border-indigo-500
                           bg-slate-800/40 hover:bg-slate-800/70 transition-all duration-300 cursor-pointer
                           flex flex-col items-center justify-center gap-5 select-none"
                id="upload-drop-zone"
              >
                <div className="relative">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-indigo-600/20 to-purple-600/20
                                  border border-indigo-500/40 flex items-center justify-center">
                    <FileAudio className="w-10 h-10 text-indigo-400" />
                  </div>
                  <motion.div
                    className="absolute -inset-2 rounded-full border border-indigo-500/20"
                    animate={{ scale: [1, 1.15], opacity: [0.6, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>
                <div className="text-center">
                  <p className="text-slate-200 font-medium text-lg">Click to select an audio file</p>
                  <p className="text-slate-500 text-sm mt-1">
                    Supports WAV · MP3 · WebM · OGG · M4A · FLAC
                  </p>
                </div>
                <div className="px-5 py-2 rounded-xl bg-indigo-600/20 border border-indigo-500/30
                                text-indigo-300 text-sm font-medium">
                  Browse Files
                </div>
              </motion.div>
            ) : (
              /* ── File Preview ────────────────────────────────────────── */
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full rounded-[2rem] border border-slate-700 bg-slate-800/60 p-6 flex flex-col gap-5"
              >
                {/* File info row */}
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-600/30 to-purple-600/30
                                  border border-indigo-500/40 flex items-center justify-center flex-shrink-0">
                    <FileAudio className="w-6 h-6 text-indigo-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-100 font-medium truncate">{uploadedFile.name}</p>
                    <p className="text-slate-500 text-sm mt-0.5">
                      {formatFileSize(uploadedFile.size)} · {uploadedFile.type || 'audio file'}
                    </p>
                  </div>
                  <button
                    onClick={clearUpload}
                    id="clear-upload-btn"
                    className="w-8 h-8 rounded-full bg-slate-700 hover:bg-red-900/60 flex items-center
                               justify-center transition-colors text-slate-400 hover:text-red-400"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Audio preview player */}
                <audio
                  controls
                  src={previewUrlRef.current || ''}
                  className="w-full h-10 rounded-xl accent-indigo-500"
                  style={{ colorScheme: 'dark' }}
                />

                {/* Submit button */}
                <Button
                  type="button"
                  id="upload-submit-btn"
                  onClick={handleUploadSubmit}
                  className="w-full h-12 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600
                             hover:from-indigo-500 hover:to-purple-500 text-white font-semibold
                             text-base shadow-lg shadow-indigo-900/30 border-0 transition-all duration-200"
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Evaluate This Recording
                </Button>

                {/* Change file link */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  id="change-file-btn"
                  className="text-slate-500 hover:text-indigo-400 text-sm text-center transition-colors"
                >
                  Choose a different file
                </button>
              </motion.div>
            )}
          </motion.div>
        )}

      </AnimatePresence>
    </motion.div>
  );
}
