import { motion } from 'framer-motion';

export default function ProcessingState() {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center space-y-12 w-full max-w-md relative">
       <div className="absolute inset-0 bg-indigo-500/10 blur-[100px] rounded-full pointer-events-none w-64 h-64" />
       
       <div className="relative flex justify-center items-center w-24 h-24">
         <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          className="absolute w-full h-full rounded-full border-2 border-t-indigo-400 border-r-purple-500 border-b-transparent border-l-transparent"
         />
         <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 shadow-inner flex justify-center items-center">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 animate-pulse" />
         </div>
       </div>

       <div className="text-center space-y-3">
         <h2 className="text-2xl font-semibold tracking-tight text-slate-100">Analyzing your response...</h2>
         <p className="text-slate-400 text-sm">Processing audio and evaluating metrics</p>
       </div>
    </motion.div>
  );
}
