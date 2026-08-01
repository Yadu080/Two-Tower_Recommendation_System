import { useState, useEffect } from 'react'

const LINKS = ['Home', 'Movies', 'New & Popular', 'My List']

/**
 * Top navigation bar. Transparent over the billboard, fading to solid as the
 * page scrolls — the usual streaming-app header treatment.
 */
export default function Navbar({ user, onSwitchProfile, onDemoClick, showDebug, onToggleDebug }) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className="fixed top-0 inset-x-0 z-50 px-6 md:px-12 py-3 flex items-center gap-6 transition-colors duration-300"
      style={{
        background: scrolled
          ? '#141414'
          : 'linear-gradient(to bottom, rgba(0,0,0,0.75) 10%, transparent 100%)',
      }}
    >
      {/* Wordmark */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="w-7 h-7 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-xs">
          R
        </div>
        <span className="text-[#E50914] font-black text-xl tracking-tight hidden sm:block">
          RECOMAI
        </span>
      </div>

      {/* Nav links — presentational, this is a single-view demo app */}
      <nav className="hidden lg:flex items-center gap-5">
        {LINKS.map((l, i) => (
          <span
            key={l}
            className={`text-sm cursor-default ${
              i === 0 ? 'text-white font-semibold' : 'text-[#e5e5e5]/80'
            }`}
          >
            {l}
          </span>
        ))}
      </nav>

      {/* Right cluster */}
      <div className="ml-auto flex items-center gap-3">
        <button
          onClick={onDemoClick}
          className="text-xs text-[#b3b3b3] hover:text-white transition-colors hidden sm:block"
        >
          Demo profiles
        </button>

        <button
          onClick={onToggleDebug}
          title="Toggle model signals"
          className={`text-xs px-2.5 py-1.5 rounded border transition-colors ${
            showDebug
              ? 'border-[#E50914] text-[#E50914]'
              : 'border-[#333] text-[#737373] hover:border-[#555]'
          }`}
        >
          ⚙
        </button>

        <button
          onClick={onSwitchProfile}
          title="Switch profile"
          className="flex items-center gap-2 rounded hover:opacity-80 transition-opacity"
        >
          <div
            className="w-8 h-8 rounded flex items-center justify-center font-bold text-white text-sm"
            style={{ background: user?.is_new ? '#E50914' : '#2f80ed' }}
          >
            {user?.name?.[0]?.toUpperCase()}
          </div>
          <span className="text-sm text-white hidden md:block">{user?.name}</span>
          <svg className="w-3 h-3 text-white hidden md:block" viewBox="0 0 12 12" fill="none">
            <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </header>
  )
}
