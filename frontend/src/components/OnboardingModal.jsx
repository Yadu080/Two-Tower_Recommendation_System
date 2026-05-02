import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchGenres, registerUser } from '../api'

const GENRE_META = {
  Action:      { emoji: '💥', color: 'border-red-500 bg-red-500/10 text-red-300' },
  Adventure:   { emoji: '🗺️', color: 'border-orange-500 bg-orange-500/10 text-orange-300' },
  Animation:   { emoji: '🎨', color: 'border-yellow-500 bg-yellow-500/10 text-yellow-300' },
  Children:    { emoji: '🧒', color: 'border-lime-500 bg-lime-500/10 text-lime-300' },
  Comedy:      { emoji: '😄', color: 'border-green-500 bg-green-500/10 text-green-300' },
  Crime:       { emoji: '🔍', color: 'border-teal-500 bg-teal-500/10 text-teal-300' },
  Documentary: { emoji: '📽️', color: 'border-cyan-500 bg-cyan-500/10 text-cyan-300' },
  Drama:       { emoji: '🎭', color: 'border-blue-500 bg-blue-500/10 text-blue-300' },
  Fantasy:     { emoji: '🧙', color: 'border-indigo-500 bg-indigo-500/10 text-indigo-300' },
  'Film-Noir': { emoji: '🌑', color: 'border-gray-400 bg-gray-500/10 text-gray-300' },
  Horror:      { emoji: '👻', color: 'border-red-700 bg-red-700/10 text-red-400' },
  IMAX:        { emoji: '🎬', color: 'border-purple-500 bg-purple-500/10 text-purple-300' },
  Musical:     { emoji: '🎵', color: 'border-pink-500 bg-pink-500/10 text-pink-300' },
  Mystery:     { emoji: '🕵️', color: 'border-violet-500 bg-violet-500/10 text-violet-300' },
  Romance:     { emoji: '❤️', color: 'border-rose-500 bg-rose-500/10 text-rose-300' },
  'Sci-Fi':    { emoji: '🚀', color: 'border-sky-500 bg-sky-500/10 text-sky-300' },
  Thriller:    { emoji: '😰', color: 'border-amber-500 bg-amber-500/10 text-amber-300' },
  War:         { emoji: '⚔️', color: 'border-stone-400 bg-stone-500/10 text-stone-300' },
  Western:     { emoji: '🤠', color: 'border-yellow-700 bg-yellow-700/10 text-yellow-500' },
}

const DEFAULT_COLOR = 'border-gray-600 bg-gray-700/10 text-gray-300'

export default function OnboardingModal({ onClose, onCreated }) {
  const [step, setStep]             = useState(1)   // 1=name, 2=genres
  const [name, setName]             = useState('')
  const [genres, setGenres]         = useState([])
  const [allGenres, setAllGenres]   = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]           = useState('')

  useEffect(() => {
    fetchGenres()
      .then(d => setAllGenres(d.genres))
      .catch(() => setAllGenres(Object.keys(GENRE_META)))
  }, [])

  const toggleGenre = (g) =>
    setGenres(prev => prev.includes(g) ? prev.filter(x => x !== g) : [...prev, g])

  const handleSubmit = async () => {
    if (genres.length === 0) { setError('Pick at least one genre'); return }
    setSubmitting(true)
    setError('')
    try {
      const user = await registerUser(name.trim(), genres)
      onCreated({ id: user.user_id, name: user.name, genres: user.genres, is_new: true })
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Something went wrong')
      setSubmitting(false)
    }
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={e => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ scale: 0.92, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.92, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-lg shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
            <div>
              <h2 className="text-lg font-bold text-white">Create your profile</h2>
              <p className="text-xs text-gray-500 mt-0.5">Step {step} of 2 · {step === 1 ? 'Your name' : 'Your tastes'}</p>
            </div>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-300 transition-colors text-xl leading-none">×</button>
          </div>

          <div className="px-6 py-5">
            <AnimatePresence mode="wait">
              {step === 1 ? (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  <p className="text-gray-400 text-sm">What should we call you?</p>
                  <input
                    autoFocus
                    type="text"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && name.trim() && setStep(2)}
                    placeholder="Your name"
                    maxLength={40}
                    className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white
                               placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors text-sm"
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-3"
                >
                  <p className="text-gray-400 text-sm">Pick the genres you enjoy — select as many as you like.</p>
                  <div className="grid grid-cols-3 gap-2 max-h-64 overflow-y-auto pr-1">
                    {allGenres.map(g => {
                      const meta = GENRE_META[g] ?? { emoji: '🎞️', color: DEFAULT_COLOR }
                      const active = genres.includes(g)
                      return (
                        <motion.button
                          key={g}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => toggleGenre(g)}
                          className={`flex items-center gap-1.5 px-2 py-2 rounded-xl border text-xs font-medium
                                      transition-all ${active ? meta.color : 'border-gray-700 text-gray-500 hover:border-gray-600 hover:text-gray-400'}`}
                        >
                          <span>{meta.emoji}</span>
                          <span className="truncate">{g}</span>
                          {active && <span className="ml-auto text-xs">✓</span>}
                        </motion.button>
                      )
                    })}
                  </div>
                  {genres.length > 0 && (
                    <p className="text-xs text-purple-400">{genres.length} selected</p>
                  )}
                  {error && <p className="text-xs text-red-400">{error}</p>}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Footer */}
          <div className="px-6 pb-5 flex items-center justify-between gap-3">
            {step === 2 ? (
              <button
                onClick={() => setStep(1)}
                className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
              >
                ← Back
              </button>
            ) : <div />}

            {step === 1 ? (
              <button
                disabled={!name.trim()}
                onClick={() => setStep(2)}
                className="ml-auto px-5 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed
                           text-white text-sm font-medium rounded-xl transition-colors"
              >
                Next →
              </button>
            ) : (
              <button
                disabled={genres.length === 0 || submitting}
                onClick={handleSubmit}
                className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed
                           text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2"
              >
                {submitting ? (
                  <>
                    <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    Creating…
                  </>
                ) : 'Get my picks →'}
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
