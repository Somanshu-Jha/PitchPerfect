import FadeIn from './FadeIn';

interface FeaturesSectionProps {
  onStartRecording: () => void;
}

/**
 * FeaturesSection - "How It Works" and structured feature grid.
 */
export default function FeaturesSection({ onStartRecording }: FeaturesSectionProps) {
  return (
    <div id="features" className="w-full py-24 bg-[#fcfdfe]/60 border-t border-slate-100/50">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <FadeIn delay={0} yOffset={20}>
          <div className="text-center space-y-4 mb-20">
            <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-slate-800 dark:text-white">
              How It Works
            </h2>
            <p className="text-lg text-slate-500 dark:text-gray-400 font-medium max-w-2xl mx-auto">
              Mastering your introduction is just three simple steps away.
            </p>
          </div>
        </FadeIn>

        {/* ── HOW IT WORKS (3-COLUMN) ── */}
        <div className="grid md:grid-cols-3 gap-10">
          {[
            {
              step: '01',
              title: 'Record your intro',
              desc: 'Click start and deliver your 30-60 second elevator pitch naturally.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 016 0v6a3 3 0 01-3 3z" />
                </svg>
              )
            },
            {
              step: '02',
              title: 'AI analyzes speech',
              desc: 'Our neural engine extracts metrics for clarity, tone, and filler words.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              )
            },
            {
              step: '03',
              title: 'Get score & feedback',
              desc: 'Receive a precision performance report with actionable improvement tips.',
              icon: (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )
            }
          ].map((item, i) => (
            <FadeIn key={i} delay={i * 150} yOffset={30} className="group">
              <div className="saas-card p-10 h-full flex flex-col gap-6 hover:-translate-y-2 transition-all duration-300">
                <div className="flex justify-between items-start">
                  <div className="w-14 h-14 rounded-2xl bg-accent/5 flex items-center justify-center text-accent ring-1 ring-accent/10 group-hover:bg-accent group-hover:text-white transition-all duration-300">
                    {item.icon}
                  </div>
                  <span className="text-3xl font-black text-slate-100 group-hover:text-accent/10 transition-colors">
                    {item.step}
                  </span>
                </div>
                <div className="space-y-3">
                  <h3 className="text-xl font-black text-slate-800 dark:text-white tracking-tight">
                    {item.title}
                  </h3>
                  <p className="text-[15px] text-slate-500 dark:text-gray-400 font-medium leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>

        {/* CTA Button */}
        <FadeIn delay={600} yOffset={20}>
          <div className="flex justify-center mt-20">
            <button
              onClick={onStartRecording}
              className="px-12 py-5 bg-white dark:bg-black border-2 border-slate-200 dark:border-white/20 text-slate-800 dark:text-white font-bold rounded-2xl hover:border-accent hover:text-accent hover:-translate-y-1 transition-all shadow-sm"
            >
              Ready to try? Start Recording
            </button>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
