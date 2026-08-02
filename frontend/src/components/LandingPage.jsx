import { useState } from 'react'
import { motion } from 'framer-motion'

// Decorative poster wall behind the hero — colored tiles that read as a shelf
// of artwork without shipping any real cover images.
const TILES = [
  '#1a1a2e','#16213e','#0f3460','#533483','#2b2d42','#3d0c11',
  '#8d0801','#540b0e','#1b1b2f','#162447','#1f4068','#2c003e',
  '#1b262c','#0a3d62','#1e3799','#4a235a','#2c003e','#40122b',
  '#1a1a2e','#16213e','#0f3460','#533483','#2b2d42','#123c69',
  '#8d0801','#540b0e','#1b1b2f','#162447','#1f4068','#22223b',
  '#1b262c','#0a3d62','#1e3799','#4a235a','#2c003e','#3a0ca3',
]

export default function LandingPage({ onNameSubmit, onDemoClick, onSignIn, onSignUp }) {
  const [name, setName] = useState('')

  const submit = () => {
    const trimmed = name.trim()
    if (trimmed) onNameSubmit(trimmed)
  }

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden" style={{ background: '#000' }}>

      {/* ── Background poster wall ────────────────────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden opacity-40 pointer-events-none">
        <div className="grid grid-cols-6 gap-1.5 h-full" style={{ gridTemplateRows: 'repeat(6, 1fr)' }}>
          {TILES.map((color, i) => (
            <div key={i} className="rounded" style={{ background: color }} />
          ))}
        </div>
      </div>

      {/* Vignette so the wall recedes behind the copy */}
      <div className="absolute inset-0 pointer-events-none"
           style={{ background: 'radial-gradient(ellipse 80% 80% at 50% 45%, rgba(0,0,0,0.92) 25%, rgba(0,0,0,0.75) 100%)' }} />
      <div className="absolute inset-0 pointer-events-none"
           style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.75) 0%, transparent 25%, transparent 70%, rgba(0,0,0,0.85) 100%)' }} />

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="relative z-10 px-6 md:px-12 py-5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-sm">R</div>
          <span className="text-[#E50914] font-black text-2xl md:text-3xl tracking-tight">RECOMAI</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onDemoClick}
            className="text-sm text-[#b3b3b3] hover:text-white transition-colors hidden sm:block"
          >
            Demo Profiles
          </button>
          <button
            onClick={onSignIn}
            className="bg-[#E50914] hover:bg-[#B81D24] text-white text-sm font-semibold rounded px-4 py-1.5 transition-colors"
          >
            Sign In
          </button>
        </div>
      </header>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 text-center py-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-2xl w-full"
        >
          <h1 className="text-white font-black text-3xl md:text-5xl lg:text-6xl leading-tight mb-4">
            Unlimited films,<br />picked just for you.
          </h1>
          <p className="text-white text-lg md:text-2xl mb-6">
            Personalised by a two-tower neural recommender.
          </p>
          <p className="text-white text-base md:text-lg mb-5">
            Ready to watch? Enter your name to build your profile.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 max-w-xl mx-auto">
            <input
              autoFocus
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Your name"
              maxLength={40}
              className="nf-input flex-1"
            />
            <button
              onClick={submit}
              disabled={!name.trim()}
              className="nf-btn text-lg md:text-2xl px-8 disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
            >
              Get Started &rsaquo;
            </button>
          </div>

          <p className="mt-6 text-sm text-[#b3b3b3]">
            <button
              onClick={onSignUp}
              className="text-white underline hover:text-[#b3b3b3] transition-colors"
            >
              Create an account
            </button>
            {' '}to keep your picks between visits, or{' '}
            <button
              onClick={onDemoClick}
              className="text-white underline hover:text-[#b3b3b3] transition-colors"
            >
              browse demo profiles
            </button>
            .
          </p>
        </motion.div>
      </main>

      {/* ── Feature strip ─────────────────────────────────────────────────── */}
      <section className="relative z-10 border-t border-[#222] bg-black/60 px-6 md:px-12 py-8">
        <div className="max-w-5xl mx-auto grid sm:grid-cols-3 gap-6 text-center">
          {[
            { t: 'Neural retrieval',  d: 'Two 128-dim towers score 10K titles in milliseconds.' },
            { t: 'Learns as you go',  d: 'Every title you open re-ranks the next set instantly.' },
            { t: 'Explainable picks', d: 'Each recommendation shows why the model chose it.' },
          ].map(f => (
            <div key={f.t}>
              <h3 className="text-white font-bold text-base mb-1">{f.t}</h3>
              <p className="text-[#b3b3b3] text-sm leading-snug">{f.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="relative z-10 text-center py-6 text-xs text-[#737373] space-y-1 bg-black">
        <p>Two-Tower Neural Retrieval · ANN Candidate Generation · GBM Re-ranking</p>
        <p>Trained on MovieLens 20M · 136K users · 10.5K films</p>
      </footer>
    </div>
  )
}
