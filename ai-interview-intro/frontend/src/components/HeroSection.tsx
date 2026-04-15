import { useEffect, useState } from 'react';
import DeviceMock from './DeviceMock';
import FadeIn from './FadeIn';

interface HeroSectionProps {
  onDiveIn: () => void;
}

export default function HeroSection({ onDiveIn }: HeroSectionProps) {

  const messages = [
    "🚀 Practice smarter.",
    "🎯 Speak better.",
    "🔥 Get hired faster.",
  ];

  const [, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % messages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <>
    <div className="fixed top-[20px] left-1/2 -translate-x-1/2 z-[9999] w-full flex justify-center pointer-events-none">

  <div className="relative w-[860px] h-[58px] rounded-full p-[1px] bg-gradient-to-r from-blue-400/40 via-purple-400/40 to-pink-400/40 shadow-[0_0_25px_rgba(99,102,241,0.25)]">

    {/* 🔥 Inner fill */}
    <div className="relative w-full h-full rounded-full bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 backdrop-blur-xl flex items-center overflow-hidden">

      {/* ✨ edge fade */}
      <div className="absolute left-0 top-0 h-full w-24 bg-gradient-to-r from-white to-transparent z-10"></div>
      <div className="absolute right-0 top-0 h-full w-24 bg-gradient-to-l from-white to-transparent z-10"></div>

      {/* 🚂 scrolling text */}
      <div className="animate-marquee flex whitespace-nowrap gap-16 px-8 text-[15px] font-semibold text-slate-800">

        <span>🚀 Speak better. Crack interviews faster.</span>
        <span>🎯 AI-powered feedback in seconds.</span>
        <span>💡 Improve clarity, confidence & structure.</span>
        <span>⚡ Results in under 30 seconds.</span>

        {/* duplicate */}
        <span>🚀 Speak better. Crack interviews faster.</span>
        <span>🎯 AI-powered feedback in seconds.</span>
        <span>💡 Improve clarity, confidence & structure.</span>
        <span>⚡ Results in under 30 seconds.</span>

      </div>

    </div>
  </div>
</div>

      {/* 🔻 HERO SECTION (unchanged) */}
      <section
        id="hero"
        className="relative min-h-screen flex items-center overflow-hidden bg-[#FFFAF0] pt-10px"
      >
        {/* Glow */}
        <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-transparent blur-2xl opacity-60 pointer-events-none" />

        {/* Background */}
        <div className="absolute top-0 right-0 w-[900px] h-[900px] bg-blue-500/5 rounded-full blur-[140px] -mr-80 -mt-80 pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[700px] h-[700px] bg-slate-400/4 rounded-full blur-[110px] -ml-40 -mb-40 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-400/[0.03] rounded-full blur-[90px] pointer-events-none" />

        <div className="relative z-10 w-full max-w-[1300px] mx-auto px-6 md:px-10 py-24 grid lg:grid-cols-[1.5fr_1fr] gap-4 lg:gap-6 items-center">

          {/* LEFT */}
          <div className="flex flex-col gap-8 lg:-ml-10 xl:-ml-16">

            <FadeIn delay={0} yOffset={20}>
              <div className="inline-flex items-center gap-2 self-start py-1.5 px-3.5 rounded-full bg-white border border-slate-200 shadow-sm">
                <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                <span className="text-[10.5px] font-black tracking-[0.08em] text-slate-500 uppercase">
                  PitchPerfect · v1.0
                </span>
              </div>
            </FadeIn>

            <div className="space-y-5">
              <FadeIn delay={150} yOffset={20}>
                <h1 className="text-[3.2rem] md:text-[4.4rem] font-black leading-[1.05] text-slate-800 tracking-tight">
                  Nail your intro<br className="hidden md:block" />
                  <span className="text-accent"> every time.</span>
                </h1>
              </FadeIn>

              <FadeIn delay={280} yOffset={20}>
                <div className="space-y-4">
                  <p className="text-[1.1rem] text-slate-500 max-w-[680px] leading-relaxed">
                    Record your interview introduction and find out exactly where
                    you're losing the listener — then fix it before the real thing.
                  </p>

                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-3">

                      <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                        <span>👥</span>
                        <span className="text-[12px] text-slate-500">
                          Built for <b className="text-slate-800">job seekers & developers</b>
                        </span>
                      </div>

                      <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                        <span>🎙️</span>
                        <span className="text-[12px] text-slate-500">
                          Works with <b className="text-slate-800">any mic</b>
                        </span>
                      </div>

                      <div className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                        <span>⚡</span>
                        <span className="text-[12px] text-slate-500">
                          Results in <b className="text-slate-800">under 30 seconds</b>
                        </span>
                      </div>

                    </div>

                    <p className="text-[13px] text-emerald-600 font-semibold">
                      ✅ Free to try — no account needed
                    </p>
                  </div>

                </div>
              </FadeIn>
            </div>

            <FadeIn delay={400} yOffset={20}>
              <div className="grid grid-cols-3 gap-2.5 max-w-[720px]">
                {[
                  {
                    title: 'Spot where you trail off',
                    desc: 'See the exact moments you lose momentum',
                  },
                  {
                    title: 'How confident you sound',
                    desc: 'Scored on tone, pace, and energy',
                  },
                  {
                    title: 'Simple tips, right away',
                    desc: 'Plain-language advice you can act on today',
                  },
                ].map((f, i) => (
                  <div
                    key={i}
                    className="group flex flex-col gap-2.5 p-3.5 rounded-2xl bg-white border border-slate-100 shadow-sm
                               hover:shadow-[0_8px_24px_rgba(37,99,235,0.1)] hover:border-blue-200/60
                               hover:-translate-y-1 hover:scale-[1.01]
                               transition-all duration-200 ease-in-out"
                  >
                    <p className="text-[12px] font-bold text-slate-700">{f.title}</p>
                    <p className="text-[10.5px] text-slate-400">{f.desc}</p>
                  </div>
                ))}
              </div>
            </FadeIn>

            <FadeIn delay={560} yOffset={20} className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={onDiveIn}
                className="group btn-primary px-9 py-3.5 text-[15px] font-bold rounded-xl"
              >
                Try your intro →
              </button>

              <button className="text-[13.5px] text-slate-400">
                See how it works →
              </button>
            </FadeIn>

          </div>

          <FadeIn
            delay={400}
            yOffset={40}
            className="relative flex items-center justify-center lg:justify-end w-full h-full mt-10 lg:mt-0"
          >
            <DeviceMock />
          </FadeIn>

        </div>
      </section>
    </>
  );
}