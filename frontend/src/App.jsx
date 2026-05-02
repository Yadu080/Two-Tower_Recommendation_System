import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import LandingPage    from './components/LandingPage'
import GenrePicker    from './components/GenrePicker'
import MovieCard      from './components/MovieCard'
import SkeletonCard   from './components/SkeletonCard'
import DemoDrawer     from './components/DemoDrawer'
import { fetchRecommendations, registerUser, logClick } from './api'

// ── view state machine ─────────────────────────────────────────────────────
//  landing → genres → recs

export default function App() {
  const [view, setView]             = useState('landing')
  const [pendingName, setPendingName] = useState('')
  const [currentUser, setCurrentUser] = useState(null)   // {id,name,genres,is_new}
  const [recs, setRecs]             = useState([])
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [showDebug, setShowDebug]   = useState(false)
  const [showDemo, setShowDemo]     = useState(false)
  const [clickedItems, setClickedItems] = useState(new Set())

  // ── load recommendations for a given user id ────────────────────────────
  const loadRecs = useCallback(async (userId) => {
    setLoading(true)
    setError(null)
    setRecs([])
    try {
      const data = await fetchRecommendations(userId, 12)
      setRecs(data.results)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Could not fetch recommendations. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  // ── Step 1: name entered on landing page ───────────────────────────────
  const handleNameSubmit = (name) => {
    setPendingName(name)
    setView('genres')
  }

  // ── Step 2: genres chosen ─────────────────────────────────────────────
  const handleGenresSubmit = async (genres) => {
    try {
      const data = await registerUser(pendingName, genres)
      const user = { id: data.user_id, name: data.name, genres: data.genres, is_new: true }
      setCurrentUser(user)
      setClickedItems(new Set())
      setView('recs')
      loadRecs(user.id)
    } catch (e) {
      // fall through — GenrePicker shows loading, reset it
      throw e
    }
  }

  // ── Demo profile selected ──────────────────────────────────────────────
  const handleDemoSelect = (user) => {
    setCurrentUser(user)
    setClickedItems(new Set())
    setShowDemo(false)
    setView('recs')
    loadRecs(user.id)
  }

  // ── Card clicked (log + refresh) ───────────────────────────────────────
  const handleCardClick = async (movie) => {
    if (clickedItems.has(movie.movie_idx)) return
    setClickedItems(prev => new Set([...prev, movie.movie_idx]))
    await logClick(currentUser.id, movie.movie_idx).catch(() => {})
    loadRecs(currentUser.id)
  }

  // ── Switch profile → back to landing ──────────────────────────────────
  const handleSwitch = () => {
    setView('landing')
    setCurrentUser(null)
    setRecs([])
    setPendingName('')
  }

  // ──────────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Demo profiles drawer — available from any view */}
      {showDemo && (
        <DemoDrawer
          onSelect={handleDemoSelect}
          onClose={() => setShowDemo(false)}
        />
      )}

      <AnimatePresence mode="wait">
        {/* ── LANDING ─────────────────────────────────────────────────── */}
        {view === 'landing' && (
          <motion.div key="landing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <LandingPage
              onNameSubmit={handleNameSubmit}
              onDemoClick={() => setShowDemo(true)}
            />
          </motion.div>
        )}

        {/* ── GENRE PICKER ────────────────────────────────────────────── */}
        {view === 'genres' && (
          <motion.div key="genres" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -40 }}>
            <GenrePicker
              name={pendingName}
              onComplete={handleGenresSubmit}
              onBack={() => setView('landing')}
            />
          </motion.div>
        )}

        {/* ── RECOMMENDATIONS ─────────────────────────────────────────── */}
        {view === 'recs' && (
          <motion.div key="recs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="min-h-screen" style={{ background: '#141414' }}>

            {/* Netflix-style header */}
            <header className="sticky top-0 z-30 px-6 md:px-12 py-3 flex items-center justify-between"
                    style={{ background: 'linear-gradient(to bottom, rgba(20,20,20,1) 70%, transparent)' }}>
              {/* Logo */}
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-xs">R</div>
                <span className="text-[#E50914] font-bold text-xl tracking-tight hidden sm:block">RECOMAI</span>
              </div>

              {/* Right controls */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowDemo(true)}
                  className="text-xs text-[#b3b3b3] hover:text-white transition-colors hidden sm:block"
                >
                  Demo profiles
                </button>
                <button
                  onClick={() => setShowDebug(v => !v)}
                  className={`text-xs px-3 py-1.5 rounded border transition-colors ${
                    showDebug ? 'border-[#E50914] text-[#E50914]' : 'border-[#333] text-[#737373] hover:border-[#555]'
                  }`}
                >
                  {showDebug ? '⚙ Debug' : '⚙'}
                </button>

                {/* Profile chip */}
                <button
                  onClick={handleSwitch}
                  title="Switch profile"
                  className="flex items-center gap-2 pl-1 pr-3 py-1 rounded-full hover:bg-white/5 transition-colors"
                >
                  <div className="w-8 h-8 rounded-full flex items-center justify-center font-bold text-white text-sm"
                       style={{ background: currentUser?.is_new ? '#E50914' : '#333' }}>
                    {currentUser?.name?.[0]?.toUpperCase()}
                  </div>
                  <span className="text-sm text-white hidden sm:block">{currentUser?.name}</span>
                </button>
              </div>
            </header>

            <main className="px-6 md:px-12 pb-16 -mt-2">
              {/* Section title */}
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    Top Picks for {currentUser?.name}
                  </h2>
                  {currentUser?.is_new && currentUser.genres?.length > 0 && (
                    <p className="text-[#737373] text-sm mt-1">
                      Based on: {currentUser.genres.join(', ')}
                    </p>
                  )}
                </div>
                <button
                  onClick={() => loadRecs(currentUser.id)}
                  className="text-xs text-[#737373] hover:text-white transition-colors flex items-center gap-1"
                >
                  ↺ Refresh
                </button>
              </div>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                              className="mb-6 p-4 rounded-lg text-sm text-red-300"
                              style={{ background: 'rgba(127,29,29,0.3)', border: '1px solid rgba(153,27,27,0.5)' }}>
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Movie grid — poster aspect ratio */}
              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
                {loading
                  ? Array.from({ length: 12 }).map((_, i) => (
                      <div key={i} className="rounded-md animate-pulse"
                           style={{ aspectRatio: '2/3', background: '#2f2f2f' }} />
                    ))
                  : recs.map((movie, i) => (
                      <motion.div
                        key={movie.movie_idx}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04 }}
                      >
                        <MovieCard
                          movie={movie}
                          onClick={handleCardClick}
                          showDebug={showDebug}
                        />
                      </motion.div>
                    ))
                }
              </div>

              {/* Clicked hint */}
              {clickedItems.size > 0 && !loading && (
                <p className="text-center text-[#737373] text-xs mt-6">
                  {clickedItems.size} film{clickedItems.size > 1 ? 's' : ''} marked as seen · recommendations updated
                </p>
              )}

              {/* Pipeline pills */}
              <div className="mt-12 flex flex-wrap justify-center gap-2">
                {['UserTower → 128-dim embedding', 'ANN top-500 retrieval', 'GBM re-rank (AUC 0.98)', 'O(N log K) heap top-12'].map(s => (
                  <span key={s} className="text-xs px-3 py-1.5 rounded-full text-[#737373]"
                        style={{ background: '#1f1f1f', border: '1px solid #2f2f2f' }}>
                    {s}
                  </span>
                ))}
              </div>
            </main>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
