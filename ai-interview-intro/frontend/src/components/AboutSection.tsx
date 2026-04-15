import FadeIn from './FadeIn';

interface AboutSectionProps {
  onStartRecording: () => void;
}

/**
 * AboutSection - "Why Choose PitchPerfect" and "Who Is It For"
 */
export default function AboutSection({ onStartRecording }: AboutSectionProps) {
  return (
    <div id="about" className="w-full py-24 bg-[#f8fafc] border-t border-slate-100/50">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <FadeIn delay={0} yOffset={20}>
          <div className="text-center space-y-4 mb-20">
            <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-slate-800">
              Why Choose PitchPerfect?
            </h2>
            <p className="text-lg text-slate-500 font-medium max-w-2xl mx-auto">
              Our AI engine delivers precision insights that go beyond simple video recording.
            </p>
          </div>
        </FadeIn>

        {/* ── WHY CHOOSE (2x2 GRID) ── */}
        <div className="grid md:grid-cols-2 gap-8 mb-32">
          {[
            {
              title: 'Real-time transcription',
              desc: 'Our engine instantly converts your speech into highly accurate text for semantic analysis.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              )
            },
            {
              title: 'AI-powered feedback',
              desc: 'Receive industry-grade feedback for each sentence you deliver in your pitch.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              )
            },
            {
              title: 'Confidence scoring',
              desc: 'A unique algorithmic approach to gauging your delivery pace and energy levels.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              )
            },
            {
              title: 'Instant performance report',
              desc: 'No more waiting. Get your dashboard instantly after the session ends.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                </svg>
              )
            }
          ].map((item, i) => (
            <FadeIn key={i} delay={i * 150} yOffset={20}>
              <div className="saas-card p-10 flex gap-6 hover:-translate-y-1 transition-all duration-300">
                <div className="w-12 h-12 rounded-xl bg-accent/5 flex items-center justify-center text-accent ring-1 ring-accent/10 flex-shrink-0">
                  {item.icon}
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-black text-slate-800 tracking-tight">
                    {item.title}
                  </h3>
                  <p className="text-[14px] text-slate-500 font-medium leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>

        {/* ── WHO IS IT FOR (3 CARDS) ── */}
        <div className="text-center space-y-4 mb-20">
          <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-slate-800">
            Who Is It For?
          </h2>
          <p className="text-lg text-slate-500 font-medium max-w-2xl mx-auto">
            A tool designed for those who aim for a flawless first impression.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-32">
          {[
            {
              title: 'Students',
              desc: 'Preparing for internships or college placements. Get that competitive edge.',
              icon: '🎓'
            },
            {
              title: 'Job Seekers',
              desc: 'Perfecting the elevator pitch before the next big interview call.',
              icon: '🚀'
            },
            {
              title: 'Developers & Freshers',
              desc: 'Software engineers honing their communication and pitch skills.',
              icon: '💻'
            }
          ].map((card, i) => (
            <FadeIn key={i} delay={i * 150} yOffset={20}>
              <div className="saas-card p-10 text-center flex flex-col items-center gap-6 group hover:-translate-y-2 transition-all duration-300">
                <div className="text-4xl grayscale group-hover:grayscale-0 transition-all duration-300">
                  {card.icon}
                </div>
                <div className="space-y-4">
                  <h3 className="text-xl font-black text-slate-800 tracking-tight">
                    {card.title}
                  </h3>
                  <p className="text-[14px] text-slate-500 font-medium leading-relaxed">
                    {card.desc}
                  </p>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>

        {/* Final CTA Strip */}
        <FadeIn delay={450} yOffset={20}>
          <div className="relative overflow-hidden rounded-[32px] bg-accent p-12 md:p-16 text-center text-white shadow-xl shadow-blue-500/10">
             <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_30%_20%,rgba(255,255,255,0.1),transparent)]" />
             <div className="relative z-10 space-y-8">
                <h3 className="text-3xl md:text-4xl font-black tracking-tight">Try it yourself</h3>
                <p className="text-blue-100 font-medium max-w-xl mx-auto opacity-80">
                  Ready to see how you perform? Record your introduction now and get instant AI feedback.
                </p>
                <button
                  onClick={onStartRecording}
                  className="px-12 py-5 bg-white text-accent font-black rounded-2xl hover:bg-blue-50 hover:-translate-y-1 transition-all shadow-lg"
                >
                  Let’s Begin
                </button>
             </div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
