import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Mic, Square, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

export default function InterviewScreen({ onFinish }: { onFinish: (blob: Blob) => void }) {
  const [isRecording, setIsRecording] = useState(false);
  const [time, setTime] = useState(0);
  const [transcript, setTranscript] = useState('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recognitionRef = useRef<any>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => setTime((t) => t + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

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
        onFinish(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        
        recognitionRef.current.onresult = (event: any) => {
          let currentTranscript = '';
          for (let i = 0; i < event.results.length; ++i) {
            currentTranscript += event.results[i][0].transcript;
          }
          setTranscript(currentTranscript);
        };
        
        recognitionRef.current.start();
      } else {
        setTranscript("Live transcription is not supported in this browser. Please speak clearly, your audio is being recorded.");
      }

      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      onFinish(new Blob());
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      if (recognitionRef.current) {
         recognitionRef.current.stop();
      }
      setIsRecording(false);
    }
  };

  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-2xl flex flex-col items-center gap-10">
      <div className="flex justify-between items-center w-full px-2">
        <h2 className="text-2xl font-semibold tracking-tight text-slate-100">Live Session</h2>
        <div className="flex items-center gap-3">
          {isRecording && <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />}
          <span className="text-2xl font-mono text-slate-400 font-medium">{formatTime(time)}</span>
        </div>
      </div>
      
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
              <p className="text-indigo-200 text-sm font-medium tracking-widest uppercase animate-pulse">Listening</p>
            </div>
            <div className="flex-1 overflow-y-auto text-lg leading-relaxed text-slate-300 italic pr-2 scrollbar-thin scrollbar-thumb-slate-700">
              {transcript ? `"${transcript}"` : "Speak now, transcription is active..."}
            </div>
            <div className="flex gap-1.5 items-end h-8 mt-4 w-full justify-center opacity-50">
              {[...Array(24)].map((_, i) => (
                <motion.div 
                  key={i} 
                  className="w-1 bg-indigo-500 rounded-full" 
                  animate={{ height: ["20%", "100%", "20%"] }} 
                  transition={{ repeat: Infinity, duration: 0.8 + Math.random(), ease: "easeInOut" }} 
                />
              ))}
            </div>
          </div>
        )}
      </Card>
      
      <div className="relative mt-4">
        <Button 
          size="lg" 
          onClick={isRecording ? stopRecording : startRecording}
          className={`h-24 w-24 rounded-full shadow-2xl transition-all duration-300 ${isRecording ? 'bg-slate-800 hover:bg-slate-700 border border-slate-600' : 'bg-gradient-to-br from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 border-0'} flex items-center justify-center`}
        >
          {isRecording ? <Square className="h-8 w-8 text-red-400" /> : <Mic className="h-10 w-10 text-white" />}
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
  );
}
