interface LaunchTryBadgeProps {
  className?: string;
}

export default function LaunchTryBadge({ className = '' }: LaunchTryBadgeProps) {
  return (
    <a
      href="https://launchtry.com/product/pitchperfect-ai?utm_source=pitchperfect-ai-lilac.vercel.app&utm_medium=badge&utm_campaign=launch_badge"
      target="_blank"
      rel="noopener noreferrer"
      aria-label="View PitchPerfect AI on LaunchTry"
      data-launchtry-badge="true"
      className={`inline-flex items-center transition-transform duration-200 hover:scale-[1.02] hover:opacity-95 ${className}`}
      style={{ display: 'inline-flex', alignItems: 'center', lineHeight: 0, textDecoration: 'none' }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="184"
        height="44"
        viewBox="0 0 184 44"
        role="img"
        aria-label="Launching on LaunchTry"
        style={{ display: 'block', width: '184px', height: '44px', maxWidth: '100%' }}
      >
        <path
          d="M14 0.5H170A13.5 13.5 0 0 1 183.5 14V43.5H0.5V14A13.5 13.5 0 0 1 14 0.5Z"
          fill="#FFFFFF"
          stroke="#E8E3DA"
        />
        <rect x="1" y="42" width="182" height="1.5" fill="#F97316" opacity="0.92" />
        <g
          transform="translate(14 10)"
          fill="none"
          stroke="#111111"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 17.8c5.2-6.9 10.9-11.6 18-14.8-1.4 7.5-4.9 13.9-10.8 19.5l-1.8-6-5.4 1.3Z" />
          <path d="M11.3 8.8 16.5 14" />
          <path d="M3.3 21.2c3.3-.1 6.3-.8 9-2.1" />
        </g>
        <text
          x="45"
          y="18"
          fill="#6B6258"
          fontFamily="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
          fontSize="9"
          fontWeight="600"
          letterSpacing=".9"
        >
          LAUNCHING ON
        </text>
        <text
          x="45"
          y="31"
          fill="#111111"
          fontFamily="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
          fontSize="15"
          fontWeight="700"
        >
          LaunchTry
        </text>
      </svg>
    </a>
  );
}
