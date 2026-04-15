/**
 * Navbar - Clean, minimal navigation for SaaS landing page.
 */

export default function Navbar({ onOpenHistory }: { onOpenHistory?: () => void }) {

  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_email");
    window.location.reload();
  };

  return (
    <nav className="fixed top-0 left-0 w-full z-[100] px-6 md:px-12 py-6 flex justify-between items-center bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-100/50 dark:border-slate-700/50 transition-colors duration-300">

      {/* Logo */}
      <div 
        className="flex items-center gap-2 cursor-pointer group"
        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      >
        <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center shadow-[0_4px_12px_rgba(37,99,235,0.2)] group-hover:scale-110 transition-transform">
          <span className="text-white font-black text-sm">PP</span>
        </div>

        <span className="text-xl font-black tracking-tighter text-slate-800 dark:text-white">
          PitchPerfect
        </span>
      </div>

      {/* Nav Items */}
      <div className="flex items-center gap-8">

        <button 
          onClick={() => scrollTo('features')}
          className="text-sm font-bold text-slate-500 dark:text-slate-300 hover:text-accent transition-colors"
        >
          Features
        </button>

        <button 
          onClick={() => scrollTo('about')}
          className="text-sm font-bold text-slate-500 dark:text-slate-300 hover:text-accent transition-colors"
        >
          About
        </button>

        <button 
          onClick={onOpenHistory}
          className="text-sm font-bold text-slate-500 dark:text-slate-300 hover:text-accent transition-colors border-r border-slate-200 dark:border-slate-700 pr-4"
        >
          History
        </button>

        <button 
          onClick={handleLogout}
          className="text-sm font-bold px-4 py-2 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-500/10 dark:text-red-400 dark:hover:bg-red-500/20 transition-all border border-red-100 dark:border-red-900 shadow-sm"
        >
          Logout
        </button>

      </div>
    </nav>
  );
}