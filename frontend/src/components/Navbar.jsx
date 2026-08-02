import { useState, useEffect } from 'react'

/**
 * Top navigation bar. Transparent over the billboard, fading to solid as the
 * page scrolls.
 *
 * Only routes that actually go somewhere appear here — decorative links that
 * did nothing when clicked were worse than having no nav at all.
 */
export default function Navbar({
  user, view, onNavigate, onSwitchProfile, onSignIn, onDemoClick,
  showDebug, onToggleDebug, myListCount = 0, isAuthed,
}) {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // close the profile menu on any outside click
  useEffect(() => {
    if (!menuOpen) return
    const close = () => setMenuOpen(false)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [menuOpen])

  const linkClass = (target) =>
    `text-sm transition-colors ${
      view === target ? 'text-white font-semibold' : 'text-[#e5e5e5]/80 hover:text-white'
    }`

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
      <button onClick={() => onNavigate('recs')} className="flex items-center gap-2 shrink-0">
        <div className="w-7 h-7 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-xs">
          R
        </div>
        <span className="text-[#E50914] font-black text-xl tracking-tight hidden sm:block">
          RECOMAI
        </span>
      </button>

      <nav className="flex items-center gap-5">
        <button onClick={() => onNavigate('recs')} className={linkClass('recs')}>
          Home
        </button>
        <button onClick={() => onNavigate('mylist')} className={linkClass('mylist')}>
          My List
          {myListCount > 0 && (
            <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full bg-[#E50914] text-white align-middle">
              {myListCount}
            </span>
          )}
        </button>
      </nav>

      {/* Right cluster */}
      <div className="ml-auto flex items-center gap-3">
        {!isAuthed && (
          <button
            onClick={onDemoClick}
            className="text-xs text-[#b3b3b3] hover:text-white transition-colors hidden sm:block"
          >
            Demo profiles
          </button>
        )}

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

        {isAuthed ? (
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setMenuOpen(v => !v) }}
              className="flex items-center gap-2 rounded hover:opacity-80 transition-opacity"
            >
              <div className="w-8 h-8 rounded flex items-center justify-center font-bold text-white text-sm"
                   style={{ background: '#E50914' }}>
                {user?.name?.[0]?.toUpperCase()}
              </div>
              <span className="text-sm text-white hidden md:block">{user?.name}</span>
              <svg className="w-3 h-3 text-white hidden md:block" viewBox="0 0 12 12" fill="none">
                <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>

            {menuOpen && (
              <div
                onClick={(e) => e.stopPropagation()}
                className="absolute right-0 mt-2 w-48 rounded overflow-hidden shadow-2xl"
                style={{ background: '#141414', border: '1px solid #333' }}
              >
                <div className="px-4 py-3 border-b" style={{ borderColor: '#333' }}>
                  <p className="text-white text-sm font-semibold truncate">{user?.name}</p>
                  <p className="text-[#737373] text-xs truncate">@{user?.username}</p>
                </div>
                <button
                  onClick={() => { setMenuOpen(false); onNavigate('genres') }}
                  className="w-full text-left px-4 py-2.5 text-sm text-[#b3b3b3] hover:bg-white/5 hover:text-white transition-colors"
                >
                  Edit my genres
                </button>
                <button
                  onClick={() => { setMenuOpen(false); onSwitchProfile() }}
                  className="w-full text-left px-4 py-2.5 text-sm text-[#b3b3b3] hover:bg-white/5 hover:text-white transition-colors"
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        ) : (
          <>
            {user && (
              <div className="w-8 h-8 rounded flex items-center justify-center font-bold text-white text-sm"
                   style={{ background: '#2f80ed' }}>
                {user?.name?.[0]?.toUpperCase()}
              </div>
            )}
            <button
              onClick={onSignIn}
              className="bg-[#E50914] hover:bg-[#B81D24] text-white text-sm font-semibold rounded px-4 py-1.5 transition-colors"
            >
              Sign In
            </button>
          </>
        )}
      </div>
    </header>
  )
}
