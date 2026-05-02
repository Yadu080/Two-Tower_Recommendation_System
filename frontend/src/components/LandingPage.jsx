import { useState } from 'react'
import { motion } from 'framer-motion'

// decorative poster grid shown behind the hero — colored tiles that mimic a movie shelf
const TILES = [
  '#1a1a2e','#16213e','#0f3460','#533483','#2b2d42',
  '#8d0801','#540b0e','#1b1b2f','#162447','#1f4068',
  '#1b262c','#0a3d62','#1e3799','#4a235a','#2c003e',
  '#1a1a2e','#16213e','#0f3460','#533483','#2b2d42',
  '#8d0801','#540b0e','#1b1b2f','#162447','#1f4068',
  '#1b262c','#0a3d62','#1e3799','#4a235a','#2c003e',
]

export default function LandingPage({ onNameSubmit, onDemoClick }) {
  const [name, setName] = useState('')

  const submit = () => {
    const trimmed = name.trim()
    if (trimmed) onNameSubmit(trimmed)
  }

  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden" style={{ background: '#141414' }}>

      {/* ── Decorative background poster grid ─────────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden opacity-20 pointer-events-none">
        <div className="grid grid-cols-6 gap-1 h-full" style={{ gridTemplateRows: 'repeat(5, 1fr)' }}>
          {TILES.map((color, i) => (
            <div key={i} className="rounded-sm" style={{ background: color }} />
          ))}
        </div>
      </div>

      {/* ── Gradient overlay so the grid fades to dark at center ──────────── */}
      <div className="absolute inset-0 pointer-events-none"
           style={{ background: 'radial-gradient(ellipse 70% 70% at 50% 50%, rgba(20,20,20,0.95) 30%, rgba(20,20,20,0.5) 100%)' }} />

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="relative z-10 px-8 md:px-16 py-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-sm">R</div>
          <span className="text-[#E50914] font-bold text-2xl tracking-tight">RECOMAI</span>
        </div>
      </header>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 text-center -mt-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-xl w-full"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white leading-tight mb-3">
            Your next favourite film<br />is waiting.
          </h1>
          <p className="text-[#b3b3b3] text-lg mb-8">
            Personalised picks powered by a neural recommendation engine.
            Enter your name to get started.
          </p>

          <div className="flex flex-col sm:flex-row gap-3">
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
              className="nf-btn disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
            >
              Get Started &rsaquo;
            </button>
          </div>

          <p className="mt-6 text-sm text-[#737373]">
            Want to explore first?{' '}
            <button
              onClick={onDemoClick}
              className="text-[#b3b3b3] underline hover:text-white transition-colors"
            >
              Browse demo profiles
            </button>
          </p>
        </motion.div>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="relative z-10 text-center pb-8 text-xs text-[#737373] space-y-1">
        <p>Two-Tower Neural Retrieval · ANN Candidate Generation · GBM Re-ranking</p>
        <p>Trained on MovieLens 25M · 162K users · 10K films</p>
      </footer>
    </div>
  )
}
