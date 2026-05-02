import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchGenres } from '../api'

const GENRE_STYLE = {
  Action:      { emoji: '💥', c1: '#7f1d1d', c2: '#b91c1c' },
  Adventure:   { emoji: '🗺️', c1: '#7c2d12', c2: '#c2410c' },
  Animation:   { emoji: '🎨', c1: '#713f12', c2: '#ca8a04' },
  Children:    { emoji: '🧸', c1: '#14532d', c2: '#15803d' },
  Comedy:      { emoji: '😄', c1: '#166534', c2: '#16a34a' },
  Crime:       { emoji: '🔫', c1: '#134e4a', c2: '#0f766e' },
  Documentary: { emoji: '📽️', c1: '#1e3a5f', c2: '#1d4ed8' },
  Drama:       { emoji: '🎭', c1: '#1e1b4b', c2: '#4338ca' },
  Fantasy:     { emoji: '🧙', c1: '#2e1065', c2: '#7c3aed' },
  'Film-Noir': { emoji: '🌑', c1: '#1c1c1c', c2: '#404040' },
  Horror:      { emoji: '👻', c1: '#450a0a', c2: '#991b1b' },
  IMAX:        { emoji: '🎬', c1: '#0c1445', c2: '#1d4ed8' },
  Musical:     { emoji: '🎵', c1: '#500724', c2: '#be185d' },
  Mystery:     { emoji: '🕵️', c1: '#2e1065', c2: '#6d28d9' },
  Romance:     { emoji: '❤️', c1: '#4c0519', c2: '#e11d48' },
  'Sci-Fi':    { emoji: '🚀', c1: '#082f49', c2: '#0284c7' },
  Thriller:    { emoji: '😰', c1: '#1c1917', c2: '#78350f' },
  War:         { emoji: '⚔️', c1: '#1c1917', c2: '#44403c' },
  Western:     { emoji: '🤠', c1: '#431407', c2: '#92400e' },
}

export default function GenrePicker({ name, onComplete, onBack }) {
  const [allGenres, setAllGenres] = useState([])
  const [selected, setSelected]   = useState([])
  const [loading, setLoading]     = useState(false)

  useEffect(() => {
    fetchGenres()
      .then(d => setAllGenres(d.genres))
      .catch(() => setAllGenres(Object.keys(GENRE_STYLE)))
  }, [])

  const toggle = (g) =>
    setSelected(prev => prev.includes(g) ? prev.filter(x => x !== g) : [...prev, g])

  const handleDone = async () => {
    if (selected.length === 0) return
    setLoading(true)
    await onComplete(selected)
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#141414' }}>

      {/* Header */}
      <header className="px-8 md:px-16 py-6 flex items-center justify-between border-b border-[#2f2f2f]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-xs">R</div>
          <span className="text-[#E50914] font-bold text-xl tracking-tight">RECOMAI</span>
        </div>
        <button onClick={onBack} className="text-sm text-[#737373] hover:text-white transition-colors">
          ← Back
        </button>
      </header>

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 py-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <h1 className="text-3xl font-bold text-white mb-2">
            What do you like to watch, {name}?
          </h1>
          <p className="text-[#b3b3b3] text-sm">
            Pick your genres — select as many as you like.
            {selected.length > 0 && (
              <span className="ml-2 text-[#E50914] font-semibold">{selected.length} selected</span>
            )}
          </p>
        </motion.div>

        {/* Genre grid */}
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-3 mb-10">
          {allGenres.map((g, i) => {
            const style = GENRE_STYLE[g] ?? { emoji: '🎞️', c1: '#1f1f1f', c2: '#374151' }
            const active = selected.includes(g)
            return (
              <motion.button
                key={g}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.025 }}
                whileTap={{ scale: 0.93 }}
                onClick={() => toggle(g)}
                className="relative aspect-square rounded-lg overflow-hidden cursor-pointer"
                style={{
                  background: `linear-gradient(135deg, ${style.c1} 0%, ${style.c2} 100%)`,
                  outline: active ? '3px solid #E50914' : '3px solid transparent',
                  outlineOffset: '2px',
                }}
              >
                {/* Emoji & label */}
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 p-2">
                  <span className="text-2xl">{style.emoji}</span>
                  <span className="text-white text-xs font-semibold text-center leading-tight">{g}</span>
                </div>

                {/* Selected overlay */}
                <AnimatePresence>
                  {active && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-black/30 flex items-start justify-end p-1.5"
                    >
                      <div className="w-5 h-5 rounded-full bg-[#E50914] flex items-center justify-center">
                        <svg className="w-3 h-3 text-white" viewBox="0 0 12 12" fill="none">
                          <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>
            )
          })}
        </div>

        {/* CTA */}
        <div className="flex justify-center">
          <button
            onClick={handleDone}
            disabled={selected.length === 0 || loading}
            className="nf-btn disabled:opacity-40 disabled:cursor-not-allowed min-w-[200px] flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Building your profile…
              </>
            ) : (
              <>See my picks &rsaquo;</>
            )}
          </button>
        </div>
      </main>
    </div>
  )
}
