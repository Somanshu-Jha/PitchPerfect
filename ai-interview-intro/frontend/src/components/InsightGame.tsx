import { useRef, useEffect, useState, useCallback } from 'react';

// ─────────────────────────────────────────────
// FIELD DETECTION ENGINE
// ─────────────────────────────────────────────
type Field =
  | 'ai'
  | 'software'
  | 'data'
  | 'cloud'
  | 'cyber'
  | 'product'
  | 'finance'
  | 'marketing'
  | 'general';

const FIELD_KEYWORDS: Record<Field, string[]> = {
  ai: ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'nlp', 'computer vision', 'llm', 'generative', 'transformer', 'pytorch', 'tensorflow', 'ml engineer'],
  software: ['software', 'developer', 'engineer', 'frontend', 'backend', 'fullstack', 'full-stack', 'react', 'angular', 'node', 'java', 'python developer', 'web developer', 'mobile developer', 'ios', 'android'],
  data: ['data science', 'data analyst', 'analytics', 'statistics', 'sql', 'tableau', 'power bi', 'pandas', 'big data', 'etl', 'data engineer', 'warehouse'],
  cloud: ['cloud', 'aws', 'azure', 'gcp', 'devops', 'kubernetes', 'docker', 'ci/cd', 'infrastructure', 'sre', 'terraform', 'ansible'],
  cyber: ['security', 'cybersecurity', 'penetration', 'firewall', 'soc', 'vulnerability', 'ethical hacking', 'infosec', 'compliance', 'threat'],
  product: ['product manager', 'ux', 'ui design', 'design', 'figma', 'user research', 'wireframe', 'prototype', 'scrum master', 'agile'],
  finance: ['finance', 'banking', 'investment', 'accounting', 'mba', 'consulting', 'strategy', 'equity', 'risk', 'portfolio', 'financial analyst'],
  marketing: ['marketing', 'seo', 'content', 'social media', 'branding', 'growth', 'campaigns', 'digital marketing', 'copywriting', 'ppc', 'email marketing'],
  general: [],
};

const FIELD_LABELS: Record<Field, string> = {
  ai: 'AI & Machine Learning',
  software: 'Software Engineering',
  data: 'Data Science & Analytics',
  cloud: 'Cloud & DevOps',
  cyber: 'Cybersecurity',
  product: 'Product & Design',
  finance: 'Finance & Business',
  marketing: 'Marketing & Growth',
  general: 'Technology & Innovation',
};

const FIELD_EMOJI: Record<Field, string> = {
  ai: '🤖', software: '💻', data: '📊', cloud: '☁️', cyber: '🔒',
  product: '🎨', finance: '💹', marketing: '📣', general: '⚡',
};

function detectField(resumeName?: string, transcript?: string): Field {
  const text = `${resumeName || ''} ${transcript || ''}`.toLowerCase();
  let bestField: Field = 'general';
  let bestScore = 0;
  for (const [field, keywords] of Object.entries(FIELD_KEYWORDS) as [Field, string[]][]) {
    if (field === 'general') continue;
    let score = 0;
    for (const kw of keywords) {
      if (text.includes(kw)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestField = field;
    }
  }
  return bestField;
}

// ─────────────────────────────────────────────
// INDUSTRY FACTS DATABASE
// ─────────────────────────────────────────────
const INDUSTRY_FACTS: Record<Field, { emoji: string; fact: string }[]> = {
  ai: [
    { emoji: '🧠', fact: 'GPT-4 was trained on ~13 trillion tokens — more words than every book ever written, combined.' },
    { emoji: '🚀', fact: "Google's Gemini 2.0 can process over 2 million tokens in a single context window — an industry first." },
    { emoji: '📈', fact: 'The global AI market is projected to reach $1.8 trillion by 2030 (Bloomberg Intelligence).' },
    { emoji: '⚡', fact: 'NVIDIA\'s Blackwell B200 GPU delivers 20 petaFLOPS of AI inference — 30x over the H100.' },
    { emoji: '🌍', fact: 'Over 80% of enterprises are now investing in AI, up from 55% just two years ago.' },
    { emoji: '🎯', fact: 'AI-assisted coding tools like GitHub Copilot write roughly 46% of code for developers using them.' },
    { emoji: '🔬', fact: 'DeepMind\'s AlphaFold has predicted the 3D structure of nearly every known protein — over 200 million.' },
    { emoji: '🏥', fact: 'AI diagnostic tools now match or exceed human doctors in detecting 14 types of cancer from medical scans.' },
    { emoji: '📡', fact: 'OpenAI\'s GPT-4o processes text, audio, and vision in a single model at 320 tokens/second.' },
    { emoji: '🤝', fact: 'Microsoft invested $13 billion in OpenAI — the largest corporate AI investment in history.' },
    { emoji: '🎓', fact: 'Stanford reports that AI systems now outperform humans on 9 out of 12 standard academic benchmarks.' },
    { emoji: '💼', fact: 'McKinsey estimates generative AI could add $4.4 trillion annually to the global economy.' },
    { emoji: '🧪', fact: 'Meta\'s Llama 3.1 405B is the largest open-source AI model ever released — fully free to use.' },
    { emoji: '🏎️', fact: 'Groq\'s LPU chip achieves 500+ tokens/second inference — 10x faster than GPU-based alternatives.' },
  ],
  software: [
    { emoji: '🦀', fact: 'Rust has been voted the "most admired programming language" for 8 consecutive years on Stack Overflow.' },
    { emoji: '⚡', fact: 'GitHub Copilot writes ~46% of all new code for developers who use it (GitHub 2024 Report).' },
    { emoji: '📦', fact: 'npm serves over 2.1 billion package downloads every single day.' },
    { emoji: '🌐', fact: 'TypeScript is now used by 78% of JavaScript developers — up from 46% in 2020.' },
    { emoji: '🏗️', fact: 'The average enterprise maintains over 250 microservices in production.' },
    { emoji: '💡', fact: 'WebAssembly now runs in all major browsers and can execute code at near-native speed.' },
    { emoji: '🐍', fact: 'Python overtook JavaScript as the #1 language on GitHub in 2024 for the first time ever.' },
    { emoji: '🔄', fact: 'Git is used by 97% of professional developers — it was created by Linus Torvalds in just 10 days.' },
    { emoji: '🌍', fact: 'There are over 28 million software developers worldwide — expected to reach 45 million by 2030.' },
    { emoji: '🧩', fact: 'VS Code is used by 74% of developers, making it the most popular IDE in the world.' },
    { emoji: '🔥', fact: 'The Linux kernel has over 30 million lines of code contributed by 20,000+ developers.' },
    { emoji: '📊', fact: 'The average developer writes 10-50 lines of production-quality code per day — quality over quantity.' },
    { emoji: '🚀', fact: 'Vercel\'s Next.js handles over 1 billion requests per week across its deployed applications.' },
    { emoji: '💻', fact: 'Stack Overflow receives 100 million monthly visitors despite the rise of AI coding assistants.' },
  ],
  data: [
    { emoji: '📊', fact: 'The world generates 402.74 million terabytes of data every single day.' },
    { emoji: '💰', fact: 'Data scientists remain the #1 most in-demand job, with a 36% growth rate predicted through 2031.' },
    { emoji: '🏢', fact: 'Companies using data-driven decision-making are 23x more likely to acquire customers.' },
    { emoji: '🔬', fact: 'Apache Spark can process petabytes of data — 100x faster than Hadoop MapReduce.' },
    { emoji: '📈', fact: 'Real-time analytics adoption has grown by 76% among Fortune 500 companies since 2022.' },
    { emoji: '🧮', fact: 'Python\'s pandas library is downloaded over 150 million times per month.' },
    { emoji: '🗄️', fact: 'Snowflake processes over 3.7 billion daily queries across its data cloud platform.' },
    { emoji: '🔍', fact: 'Only 2% of the world\'s data is actually analyzed — the rest is uncategorized "dark data."' },
    { emoji: '⏱️', fact: 'Poor data quality costs organizations an average of $12.9 million per year (Gartner).' },
    { emoji: '🧊', fact: 'Apache Iceberg is replacing Hive as the standard table format for massive-scale data lakes.' },
    { emoji: '📉', fact: '73% of corporate data goes unused for analytics — most enterprises are sitting on a goldmine.' },
    { emoji: '🏆', fact: 'Kaggle has over 15 million registered data scientists competing in ML challenges worldwide.' },
  ],
  cloud: [
    { emoji: '☁️', fact: 'AWS, Azure, and GCP together control 66% of the global cloud infrastructure market.' },
    { emoji: '🐳', fact: 'Over 300 million Docker images are pulled every single day from Docker Hub.' },
    { emoji: '⚙️', fact: 'Kubernetes orchestrates containers for 96% of organizations adopting cloud-native architecture.' },
    { emoji: '📡', fact: 'Serverless computing reduces infrastructure costs by up to 77% for event-driven workloads.' },
    { emoji: '🌍', fact: 'AWS has 33 launched regions globally, with 105 availability zones.' },
    { emoji: '🚀', fact: 'Infrastructure as Code adoption grew by 300% in the last 3 years (HashiCorp survey).' },
    { emoji: '💸', fact: 'Global cloud spending exceeded $600 billion in 2024, growing 21% year-over-year.' },
    { emoji: '🔗', fact: 'Multi-cloud strategies are used by 87% of enterprises to avoid vendor lock-in.' },
    { emoji: '🏗️', fact: 'Terraform has been downloaded over 300 million times and manages infrastructure at 85% of Fortune 500.' },
    { emoji: '📦', fact: 'AWS Lambda executes over 1 trillion function invocations per month across all customers.' },
    { emoji: '🛡️', fact: 'Cloud misconfigurations are the #1 cause of cloud security breaches — ahead of all attack vectors.' },
    { emoji: '⚡', fact: 'Edge computing reduces latency to under 5ms — enabling real-time AR, gaming, and autonomous vehicles.' },
  ],
  cyber: [
    { emoji: '🔐', fact: 'A ransomware attack occurs every 11 seconds globally — up from every 40 seconds in 2016.' },
    { emoji: '💰', fact: 'The average cost of a data breach in 2024 is $4.88 million (IBM Security Report).' },
    { emoji: '🛡️', fact: 'Zero-trust architecture adoption has jumped from 16% to 61% among enterprises in 3 years.' },
    { emoji: '🕵️', fact: 'The cybersecurity workforce gap stands at 3.4 million unfilled positions worldwide.' },
    { emoji: '⚠️', fact: '95% of cybersecurity breaches are caused by human error (World Economic Forum).' },
    { emoji: '🔑', fact: 'Passkeys are replacing passwords — Google, Apple, and Microsoft now support passwordless auth.' },
    { emoji: '🌐', fact: 'DDoS attacks peaked at 3.47 Tbps in 2024 — enough to stream 800,000 HD videos simultaneously.' },
    { emoji: '🤖', fact: 'AI-powered cyberattacks increased 135% in 2024, using deepfakes for social engineering.' },
    { emoji: '🏛️', fact: 'The US government spends over $18 billion annually on federal cybersecurity programs.' },
    { emoji: '🔒', fact: 'Quantum-resistant encryption standards (NIST PQC) were finalized in 2024 for post-quantum security.' },
    { emoji: '📱', fact: 'Mobile malware infections rose 64% in 2024, with banking trojans being the fastest growing category.' },
    { emoji: '🎯', fact: 'Bug bounty programs paid out over $45 million in 2024 — HackerOne alone has 1M+ ethical hackers.' },
  ],
  product: [
    { emoji: '🎨', fact: 'Figma was acquired by Adobe for $20B — the largest private software acquisition in history.' },
    { emoji: '📱', fact: 'Users form an opinion about a website in just 0.05 seconds — first impressions are everything.' },
    { emoji: '🧪', fact: 'A/B testing can increase conversion rates by up to 49% when properly implemented.' },
    { emoji: '♿', fact: 'Web accessibility compliance (WCAG) is now legally required in 40+ countries.' },
    { emoji: '📐', fact: 'Design systems reduce UI development time by 30-50% across large organizations.' },
    { emoji: '🎯', fact: 'Products built with user research are 2.5x more likely to achieve product-market fit.' },
    { emoji: '📊', fact: 'Only 1 in 7 product features is used regularly — the rest are ignored by 80%+ of users.' },
    { emoji: '🔄', fact: 'Companies shipping weekly iterations grow 2.4x faster than those on monthly release cycles.' },
    { emoji: '🧠', fact: 'The average user\'s attention span is 8 seconds — shorter than a goldfish\'s at 9 seconds.' },
    { emoji: '💡', fact: 'Apple\'s design team had only 100 people when they created the iPhone — now it\'s over 500.' },
    { emoji: '🌍', fact: 'Dark mode is preferred by 81% of smartphone users — up from 55% in 2020.' },
    { emoji: '⚡', fact: 'A 1-second delay in page load time reduces conversions by 7% (Akamai research).' },
  ],
  finance: [
    { emoji: '📈', fact: 'Algorithmic trading accounts for over 60-73% of all US equity trading volume.' },
    { emoji: '🏦', fact: 'Global fintech investment reached $113.7 billion in 2023 across 4,547 deals.' },
    { emoji: '💎', fact: 'Warren Buffett\'s Berkshire Hathaway has returned 3,787,464% since 1964 — that\'s not a typo.' },
    { emoji: '🌍', fact: 'Over 2 billion adults worldwide still lack access to formal financial services.' },
    { emoji: '₿', fact: 'Bitcoin\'s blockchain processes ~$10 billion in daily transactions across the global network.' },
    { emoji: '📊', fact: 'ESG-focused funds now manage over $35 trillion in assets globally.' },
    { emoji: '🤖', fact: 'JPMorgan\'s AI model LOXM executes equity trades more efficiently than any human trader.' },
    { emoji: '💳', fact: 'Digital payments processed over $11.6 trillion globally in 2024 — surpassing cash for the first time.' },
    { emoji: '📉', fact: 'The 2008 financial crisis wiped out $17 trillion in household wealth in the US alone.' },
    { emoji: '🏆', fact: 'Renaissance Technologies\' Medallion Fund averaged 66% annual returns for 30 years — the best ever.' },
    { emoji: '🔗', fact: 'DeFi (Decentralized Finance) protocols manage over $90 billion in total value locked (TVL).' },
    { emoji: '⚡', fact: 'High-frequency trading firms execute trades in under 13 microseconds — faster than a human blink.' },
  ],
  marketing: [
    { emoji: '📱', fact: 'Short-form video (TikTok/Reels) generates 2.5x more engagement than any other content format.' },
    { emoji: '📧', fact: 'Email marketing returns $36 for every $1 spent — the highest ROI of any marketing channel.' },
    { emoji: '🔍', fact: 'Google processes over 8.5 billion searches every day — that\'s 99,000+ per second.' },
    { emoji: '🎯', fact: 'Personalized marketing campaigns see 5-8x higher ROI than generic broadcasts.' },
    { emoji: '📊', fact: 'Content marketing generates 3x more leads than outbound marketing at 62% lower cost.' },
    { emoji: '🤖', fact: 'AI-generated content creation tools are used by 67% of marketers as of 2024.' },
    { emoji: '📺', fact: 'YouTube Shorts reached 70 billion daily views in 2024 — more than Instagram and TikTok combined.' },
    { emoji: '🛒', fact: 'Social commerce sales hit $1.2 trillion globally in 2024 — growing 3x faster than traditional e-commerce.' },
    { emoji: '🎙️', fact: 'Podcast advertising revenue exceeded $4 billion in 2024 with a 57% YoY growth rate.' },
    { emoji: '🧲', fact: 'Interactive content (quizzes, polls) gets 52.6% more engagement than static posts.' },
    { emoji: '🌐', fact: 'Influencer marketing industry surpassed $21 billion in 2024 — 10x what it was in 2016.' },
    { emoji: '💬', fact: 'WhatsApp Business is used by 50 million companies worldwide for customer engagement.' },
  ],
  general: [
    { emoji: '⚡', fact: 'The global tech industry is worth $5.3 trillion — larger than any country\'s GDP except the US and China.' },
    { emoji: '🌐', fact: 'There are now over 5.35 billion internet users worldwide — 66% of the world\'s population.' },
    { emoji: '📱', fact: 'The average person spends 6 hours and 58 minutes online every day.' },
    { emoji: '🚀', fact: 'SpaceX\'s Starlink satellite constellation provides internet via 6,000+ satellites in orbit.' },
    { emoji: '🧬', fact: 'CRISPR gene editing can now correct genetic mutations in living patients — first approved in 2023.' },
    { emoji: '🔋', fact: 'Global EV sales surpassed 17 million units in 2024 — a 25% year-over-year increase.' },
    { emoji: '🏭', fact: 'TSMC manufactures 90% of the world\'s most advanced semiconductor chips (under 7nm).' },
    { emoji: '🕶️', fact: 'Apple Vision Pro uses 23 million pixels — more than a 4K TV for each eye.' },
    { emoji: '🤖', fact: 'Boston Dynamics\' Atlas robot can now perform parkour, backflips, and autonomous warehouse work.' },
    { emoji: '🧪', fact: 'Quantum computers achieved "quantum advantage" — solving problems impossible for classical computers.' },
    { emoji: '🛰️', fact: 'NASA\'s James Webb Space Telescope captures images from 13.4 billion years ago — near the Big Bang.' },
    { emoji: '💻', fact: 'Apple\'s M4 chip delivers 38 TOPS of neural engine performance — rivaling dedicated AI accelerators.' },
  ],
};

// ─────────────────────────────────────────────
// GAME CONSTANTS (Chrome Dino-accurate pacing)
// ─────────────────────────────────────────────
const CANVAS_W = 800;
const CANVAS_H = 250;
const GROUND_Y = 200;
const GRAVITY = 0.5;
const JUMP_FORCE = -10;
const INITIAL_SPEED = 2;        // Very gentle start — just like Chrome Dino
const MAX_SPEED = 9;            // Top speed after ~5 min of play
const SPEED_INCREMENT = 0.0003; // Takes ~3 min to reach speed 5
const OBSTACLE_INTERVAL_MIN = 90;
const OBSTACLE_INTERVAL_MAX = 200;

// ─────────────────────────────────────────────
// GAME COMPONENT
// ─────────────────────────────────────────────
interface InsightGameProps {
  resumeHint?: string;
  transcriptHint?: string;
  analysisComplete?: boolean;
  onViewResults?: () => void;
}

export default function InsightGame({
  resumeHint,
  transcriptHint,
  analysisComplete = false,
  onViewResults,
}: InsightGameProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameLoopRef = useRef<number>(0);
  const [gameState, setGameState] = useState<'ready' | 'playing' | 'gameover'>('ready');
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(() => {
    const saved = localStorage.getItem('insight_game_highscore');
    return saved ? parseInt(saved) : 0;
  });
  const [showFacts, setShowFacts] = useState(false);

  const detectedField = detectField(resumeHint, transcriptHint);
  const facts = INDUSTRY_FACTS[detectedField];

  // Random facts — re-shuffled on every game over
  const [displayFacts, setDisplayFacts] = useState<{ emoji: string; fact: string }[]>([]);

  // Fisher-Yates shuffle for true randomization
  const pickRandomFacts = useCallback(() => {
    const pool = [...INDUSTRY_FACTS[detectedField]];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, 3);
  }, [detectedField]);

  // Game state refs (mutable during animation loop)
  const stateRef = useRef({
    playerY: GROUND_Y,
    playerVY: 0,
    isJumping: false,
    obstacles: [] as { x: number; w: number; h: number; type: 'small' | 'tall' | 'double' }[],
    speed: INITIAL_SPEED,
    score: 0,
    frame: 0,
    nextObstacleIn: 120,
    cloudX: [100, 300, 550, 700],
    cloudY: [40, 70, 30, 90],
    groundOffset: 0,
    playing: false,
    duckFrame: 0,
  });

  const jump = useCallback(() => {
    const s = stateRef.current;
    if (!s.isJumping && s.playing) {
      s.playerVY = JUMP_FORCE;
      s.isJumping = true;
    }
  }, []);

  const startGame = useCallback(() => {
    const s = stateRef.current;
    s.playerY = GROUND_Y;
    s.playerVY = 0;
    s.isJumping = false;
    s.obstacles = [];
    s.speed = INITIAL_SPEED;
    s.score = 0;
    s.frame = 0;
    s.nextObstacleIn = 120;
    s.groundOffset = 0;
    s.playing = true;
    s.duckFrame = 0;
    setScore(0);
    setShowFacts(false);
    setGameState('playing');
  }, []);

  const endGame = useCallback(() => {
    const s = stateRef.current;
    s.playing = false;
    setGameState('gameover');
    const finalScore = s.score;
    setScore(finalScore);
    if (finalScore > highScore) {
      setHighScore(finalScore);
      localStorage.setItem('insight_game_highscore', String(finalScore));
    }
    // Re-randomize facts on every game over
    setDisplayFacts(pickRandomFacts());
    setTimeout(() => setShowFacts(true), 600);
  }, [highScore, pickRandomFacts]);

  // ── Drawing helpers ──
  const drawPlayer = (ctx: CanvasRenderingContext2D, y: number, frame: number) => {
    const x = 60;
    const isDark = document.documentElement.classList.contains('dark');
    const bodyColor = isDark ? '#a5b4fc' : '#4f46e5';
    const accentColor = isDark ? '#818cf8' : '#6366f1';

    // Body
    ctx.fillStyle = bodyColor;
    ctx.beginPath();
    ctx.roundRect(x, y - 36, 24, 30, 6);
    ctx.fill();

    // Head
    ctx.fillStyle = accentColor;
    ctx.beginPath();
    ctx.arc(x + 12, y - 42, 10, 0, Math.PI * 2);
    ctx.fill();

    // Eye
    ctx.fillStyle = isDark ? '#000' : '#fff';
    ctx.beginPath();
    ctx.arc(x + 16, y - 44, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = isDark ? '#a5b4fc' : '#1e1b4b';
    ctx.beginPath();
    ctx.arc(x + 17, y - 44, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Legs (animated run cycle)
    const legPhase = Math.floor(frame / 6) % 2;
    ctx.fillStyle = bodyColor;
    if (y < GROUND_Y) {
      // Jumping - legs tucked
      ctx.fillRect(x + 4, y - 6, 6, 8);
      ctx.fillRect(x + 14, y - 6, 6, 8);
    } else {
      // Running
      ctx.fillRect(x + 4, y - 6, 6, legPhase ? 10 : 6);
      ctx.fillRect(x + 14, y - 6, 6, legPhase ? 6 : 10);
    }

    // Briefcase (interview theme!)
    ctx.fillStyle = isDark ? '#fbbf24' : '#d97706';
    ctx.beginPath();
    ctx.roundRect(x + 24, y - 24, 10, 8, 2);
    ctx.fill();
    ctx.fillStyle = isDark ? '#f59e0b' : '#b45309';
    ctx.fillRect(x + 27, y - 27, 4, 3);
  };

  const drawObstacle = (ctx: CanvasRenderingContext2D, obs: { x: number; w: number; h: number; type: string }) => {
    const isDark = document.documentElement.classList.contains('dark');

    if (obs.type === 'double') {
      // Double obstacles (two small ones close together)
      ctx.fillStyle = isDark ? '#f87171' : '#dc2626';
      ctx.beginPath();
      ctx.roundRect(obs.x, GROUND_Y - obs.h, obs.w * 0.4, obs.h, 3);
      ctx.fill();
      ctx.beginPath();
      ctx.roundRect(obs.x + obs.w * 0.6, GROUND_Y - obs.h * 0.7, obs.w * 0.4, obs.h * 0.7, 3);
      ctx.fill();
    } else {
      // Single obstacle (like cactus shapes)
      ctx.fillStyle = isDark ? '#f87171' : '#dc2626';
      ctx.beginPath();
      ctx.roundRect(obs.x, GROUND_Y - obs.h, obs.w, obs.h, 4);
      ctx.fill();
      // Detail lines
      ctx.fillStyle = isDark ? '#fca5a5' : '#b91c1c';
      ctx.fillRect(obs.x + obs.w / 2 - 1, GROUND_Y - obs.h + 4, 2, obs.h - 8);
    }
  };

  // ── Main game loop ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const s = stateRef.current;
      const isDark = document.documentElement.classList.contains('dark');

      // Clear
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

      // Sky gradient
      const skyGrad = ctx.createLinearGradient(0, 0, 0, GROUND_Y);
      if (isDark) {
        skyGrad.addColorStop(0, '#0a0a0a');
        skyGrad.addColorStop(1, '#1a1a2e');
      } else {
        skyGrad.addColorStop(0, '#f8fafc');
        skyGrad.addColorStop(1, '#e2e8f0');
      }
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, CANVAS_W, GROUND_Y);

      // Clouds
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(148,163,184,0.2)';
      for (let i = 0; i < s.cloudX.length; i++) {
        const cx = s.cloudX[i];
        const cy = s.cloudY[i];
        ctx.beginPath();
        ctx.ellipse(cx, cy, 30, 10, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(cx + 15, cy - 5, 20, 8, 0, 0, Math.PI * 2);
        ctx.fill();
      }

      // Ground
      ctx.fillStyle = isDark ? '#27272a' : '#cbd5e1';
      ctx.fillRect(0, GROUND_Y, CANVAS_W, 2);

      // Ground texture (dashed lines)
      ctx.setLineDash([8, 12]);
      ctx.strokeStyle = isDark ? '#3f3f46' : '#94a3b8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(-s.groundOffset % 20, GROUND_Y + 10);
      ctx.lineTo(CANVAS_W, GROUND_Y + 10);
      ctx.stroke();
      ctx.setLineDash([]);

      if (s.playing) {
        s.frame++;
        s.duckFrame++;
        s.score = Math.floor(s.frame / 6);
        setScore(s.score);

        // Speed up
        s.speed = Math.min(MAX_SPEED, INITIAL_SPEED + s.frame * SPEED_INCREMENT);

        // ── Physics ──
        s.playerVY += GRAVITY;
        s.playerY += s.playerVY;
        if (s.playerY >= GROUND_Y) {
          s.playerY = GROUND_Y;
          s.playerVY = 0;
          s.isJumping = false;
        }

        // ── Move clouds ──
        for (let i = 0; i < s.cloudX.length; i++) {
          s.cloudX[i] -= s.speed * 0.2;
          if (s.cloudX[i] < -50) {
            s.cloudX[i] = CANVAS_W + 50 + Math.random() * 100;
            s.cloudY[i] = 20 + Math.random() * 80;
          }
        }

        // ── Ground scroll ──
        s.groundOffset += s.speed;

        // ── Spawn obstacles ──
        s.nextObstacleIn--;
        if (s.nextObstacleIn <= 0) {
          const types: ('small' | 'tall' | 'double')[] = ['small', 'tall', 'double'];
          const type = types[Math.floor(Math.random() * types.length)];
          const h = type === 'tall' ? 45 + Math.random() * 15 : 25 + Math.random() * 10;
          const w = type === 'double' ? 35 + Math.random() * 10 : 16 + Math.random() * 8;
          s.obstacles.push({ x: CANVAS_W + 20, w, h, type });
          s.nextObstacleIn = OBSTACLE_INTERVAL_MIN + Math.floor(Math.random() * (OBSTACLE_INTERVAL_MAX - OBSTACLE_INTERVAL_MIN));
        }

        // ── Move & draw obstacles ──
        s.obstacles = s.obstacles.filter((o) => o.x > -50);
        for (const obs of s.obstacles) {
          obs.x -= s.speed;
          drawObstacle(ctx, obs);

          // Collision check (hitbox slightly forgiving)
          const px = 60;
          const py = s.playerY;
          const pw = 24;
          const ph = 36;
          const margin = 6;
          if (
            px + pw - margin > obs.x &&
            px + margin < obs.x + obs.w &&
            py - ph + margin < GROUND_Y &&
            py > GROUND_Y - obs.h - margin
          ) {
            // Only collide if player is near ground level
            if (py > GROUND_Y - obs.h + margin) {
              endGame();
            }
          }
        }
      }

      // Draw player
      drawPlayer(ctx, s.playerY, s.duckFrame);

      // Score display on canvas
      ctx.fillStyle = isDark ? '#a1a1aa' : '#64748b';
      ctx.font = 'bold 14px "Inter", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`SCORE: ${String(s.score).padStart(5, '0')}`, CANVAS_W - 20, 30);

      if (s.playing) {
        gameLoopRef.current = requestAnimationFrame(render);
      }
    };

    // Initial render
    render();

    return () => {
      cancelAnimationFrame(gameLoopRef.current);
    };
  }, [gameState, endGame]);

  // ── Keyboard & Touch Input ──
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.code === 'Space' || e.code === 'ArrowUp') {
        e.preventDefault();
        if (gameState === 'ready' || gameState === 'gameover') {
          startGame();
        } else if (gameState === 'playing') {
          jump();
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [gameState, startGame, jump]);

  const handleCanvasClick = () => {
    if (gameState === 'ready' || gameState === 'gameover') {
      startGame();
    } else if (gameState === 'playing') {
      jump();
    }
  };

  // ── Auto-transition when analysis completes ──
  useEffect(() => {
    if (analysisComplete && gameState !== 'gameover') {
      // If playing, let them finish. If ready, auto-transition.
      if (gameState === 'ready') {
        onViewResults?.();
      }
    }
  }, [analysisComplete, gameState, onViewResults]);

  return (
    <section className="min-h-screen py-16 flex flex-col items-center justify-center w-full px-4">
      <div className="w-full max-w-3xl flex flex-col items-center gap-8">

        {/* Header */}
        <div className="text-center space-y-3 animate-fade-in-up">
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
            <span className="text-xs font-black uppercase tracking-[0.25em] text-accent">
              Analyzing Your Interview
            </span>
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
          </div>
          <p className="text-sm text-slate-500 dark:text-gray-400 font-medium">
            While we extract your insights, enjoy a quick game!
          </p>
        </div>

        {/* Game Container */}
        <div className="relative w-full max-w-[800px]">
          {/* Glow effect */}
          <div className="absolute -inset-4 bg-accent opacity-[0.03] blur-3xl rounded-[40px]" />

          <div className="relative saas-card overflow-hidden bg-white dark:bg-[#0a0a0a] border border-slate-200 dark:border-white/10 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.06)]">

            {/* Canvas */}
            <div className="relative cursor-pointer" onClick={handleCanvasClick}>
              <canvas
                ref={canvasRef}
                width={CANVAS_W}
                height={CANVAS_H}
                className="w-full h-auto block"
                style={{ imageRendering: 'pixelated' }}
              />

              {/* Ready overlay */}
              {gameState === 'ready' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 dark:bg-black/80 backdrop-blur-sm">
                  <div className="flex flex-col items-center gap-4 animate-fade-in-up">
                    <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
                      <span className="text-3xl">🎮</span>
                    </div>
                    <h3 className="text-xl font-black text-slate-800 dark:text-white tracking-tight">
                      Industry Runner
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-gray-400 font-medium text-center max-w-xs">
                      Jump over obstacles while your insights are being extracted!
                    </p>
                    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-accent/5 border border-accent/20">
                      <span className="text-accent text-xs font-bold">Press SPACE or TAP to start</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Game Over overlay */}
              {gameState === 'gameover' && !showFacts && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-black/80 backdrop-blur-sm">
                  <div className="flex flex-col items-center gap-3 animate-fade-in-up">
                    <span className="text-4xl">💥</span>
                    <h3 className="text-2xl font-black text-slate-800 dark:text-white">Game Over</h3>
                    <p className="text-sm text-slate-500 dark:text-gray-400 font-bold">
                      Score: {score} {score > 0 && score >= highScore && '🏆 New High Score!'}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Score bar below canvas */}
            <div className="flex items-center justify-between px-5 py-3 border-t border-slate-100 dark:border-white/10 bg-slate-50/50 dark:bg-white/[0.02]">
              <div className="flex items-center gap-4">
                <span className="text-xs font-bold text-slate-400 dark:text-gray-500 uppercase tracking-wider">
                  Score: <span className="text-slate-700 dark:text-white tabular-nums">{String(score).padStart(5, '0')}</span>
                </span>
                <span className="text-xs font-bold text-slate-400 dark:text-gray-500 uppercase tracking-wider">
                  Best: <span className="text-accent tabular-nums">{String(highScore).padStart(5, '0')}</span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-lg">{FIELD_EMOJI[detectedField]}</span>
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400">
                  {FIELD_LABELS[detectedField]}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Facts Panel (after game over) */}
        {showFacts && (
          <div className="w-full max-w-[800px] space-y-4 insight-facts-enter">
            <div className="flex items-center gap-3 mb-2">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-accent/20 to-transparent" />
              <span className="text-xs font-black uppercase tracking-[0.2em] text-accent">
                {FIELD_EMOJI[detectedField]} {FIELD_LABELS[detectedField]} — Did You Know?
              </span>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-accent/20 to-transparent" />
            </div>

            {displayFacts.map((f, i) => (
              <div
                key={i}
                className="insight-fact-card saas-card p-5 flex items-start gap-4 bg-white dark:bg-[#0a0a0a] border border-slate-100 dark:border-white/10"
                style={{ animationDelay: `${i * 150}ms` }}
              >
                <span className="text-2xl flex-shrink-0 mt-0.5">{f.emoji}</span>
                <p className="text-sm text-slate-700 dark:text-gray-300 font-medium leading-relaxed">
                  {f.fact}
                </p>
              </div>
            ))}

            {/* Play Again */}
            <div className="flex justify-center pt-2">
              <button
                onClick={startGame}
                className="px-5 py-2.5 text-sm font-bold text-slate-500 dark:text-gray-400 bg-white dark:bg-black border border-slate-200 dark:border-white/20 rounded-xl hover:bg-slate-50 hover:text-accent hover:border-accent/40 transition-all shadow-sm"
              >
                🎮 Play Again
              </button>
            </div>
          </div>
        )}

        {/* Analysis Complete Banner */}
        {analysisComplete && (
          <div className="w-full max-w-[800px] animate-fade-in-up">
            <button
              onClick={onViewResults}
              className="w-full py-5 rounded-2xl bg-gradient-to-r from-accent to-indigo-600 text-white font-black text-lg tracking-tight shadow-[0_10px_40px_rgba(37,99,235,0.3)] hover:shadow-[0_15px_50px_rgba(37,99,235,0.4)] hover:-translate-y-1 transition-all duration-300 results-ready-glow"
            >
              ✨ Your Insights Are Ready — View Results
            </button>
          </div>
        )}

        {/* Processing indicator */}
        {!analysisComplete && (
          <div className="flex items-center gap-3 animate-pulse">
            <div className="w-2 h-2 rounded-full bg-accent animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 rounded-full bg-accent animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 rounded-full bg-accent animate-bounce" style={{ animationDelay: '300ms' }} />
            <span className="text-xs font-bold text-slate-400 dark:text-gray-500 uppercase tracking-widest ml-2">
              Extracting insights...
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
