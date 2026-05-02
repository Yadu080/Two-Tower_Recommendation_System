import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import MovieCard from './components/MovieCard'
import SkeletonCard from './components/SkeletonCard'
import DebugPanel from './components/DebugPanel'
import { fetchRecommendations, fetchUsers, logClick } from './api'

export default function App() {
  const [users, setUsers]             = useState([])
  const [selectedUser, setSelectedUser] = useState(null)
  const [recs, setRecs]               = useState([])
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [showDebug, setShowDebug]     = useState(false)
  const [cacheStats, setCacheStats]   = useState(null)
  const [latencyMs, setLatencyMs]     = useState(null)
  const [clickedItems, setClickedItems] = useState(new Set())

  // load sample users on mount
  useEffect(() => {
    fetchUsers(40)
      .then(d => {
        setUsers(d.users)
        setSelectedUser(d.users[0])
      })
      .catch(() => setError('Backend not running. Start with: uvicorn backend.main:app --reload'))
  }, [])

  // fetch recommendations when user changes
  const loadRecs = useCallback(async (userId) => {
    if (!userId) return
    setLoading(true)
    setError(null)
    setRecs([])
    try {
      const data = await fetchRecommendations(userId, 12)
      setRecs(data.results)
      setCacheStats(data.cache_stats)
      setLatencyMs(data.results[0]?.latency_ms ?? null)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Failed to fetch recommendations')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedUser) loadRecs(selectedUser)
  }, [selectedUser, loadRecs])

  const handleCardClick = async (movie) => {
    if (clickedItems.has(movie.movie_idx)) return
    setClickedItems(prev => new Set([...prev, movie.movie_idx]))
    await logClick(selectedUser, movie.movie_idx).catch(() => {})
    // refresh recommendations after click
    loadRecs(selectedUser)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between sticky top-0 bg-gray-950/90 backdrop-blur z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center text-sm font-bold">R</div>
          <span className="font-bold text-lg text-white">RecomAI</span>
          <span className="text-xs text-gray-500 hidden sm:block">Two-Tower · FAISS · LightGBM</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowDebug(v => !v)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              showDebug
                ? 'bg-purple-600/20 border-purple-500 text-purple-300'
                : 'border-gray-700 text-gray-400 hover:border-gray-600'
            }`}
          >
            {showDebug ? '⚙ Debug ON' : '⚙ Debug'}
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        {/* User selector */}
        <section>
          <h2 className="text-sm text-gray-500 uppercase tracking-wider mb-3">Select User</h2>
          <div className="flex flex-wrap gap-2">
            {users.map(uid => (
              <button
                key={uid}
                onClick={() => setSelectedUser(uid)}
                className={`text-xs px-3 py-1.5 rounded-full border transition-all ${
                  selectedUser === uid
                    ? 'bg-purple-600 border-purple-500 text-white'
                    : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300'
                }`}
              >
                User {uid}
              </button>
            ))}
          </div>
        </section>

        {/* Debug panel */}
        <DebugPanel cacheStats={cacheStats} latencyMs={latencyMs} visible={showDebug} />

        {/* Feed header */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-xl font-bold text-white">
                {selectedUser ? `Recommendations for User ${selectedUser}` : 'Select a user'}
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">
                {!loading && recs.length > 0
                  ? `${recs.length} results · click any card to update recommendations`
                  : 'Retrieval: FAISS top-500 → LightGBM rank → top-12'}
              </p>
            </div>
            {!loading && recs.length > 0 && (
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => loadRecs(selectedUser)}
                className="text-xs px-3 py-1.5 bg-gray-800 border border-gray-700
                           rounded-lg text-gray-400 hover:text-white hover:border-gray-600 transition-colors"
              >
                ↺ Refresh
              </motion.button>
            )}
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-red-950/50 border border-red-800 rounded-xl p-4 text-red-300 text-sm mb-4"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {loading
              ? Array.from({ length: 12 }).map((_, i) => <SkeletonCard key={i} />)
              : recs.map(movie => (
                  <MovieCard
                    key={movie.movie_idx}
                    movie={movie}
                    onClick={handleCardClick}
                    showDebug={showDebug}
                  />
                ))
            }
          </div>
        </section>

        {/* How it works footer */}
        <section className="border-t border-gray-800 pt-6">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">How it works</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { step: '1', label: 'User Tower', desc: 'Maps user → 128-dim vector' },
              { step: '2', label: 'FAISS ANN', desc: 'Retrieves top-500 similar items' },
              { step: '3', label: 'LightGBM', desc: 'Re-ranks by rich features' },
              { step: '4', label: 'Top-K Heap', desc: 'O(N log K) final selection' },
            ].map(s => (
              <div key={s.step} className="bg-gray-900 rounded-xl p-3 border border-gray-800">
                <div className="text-purple-400 text-xs font-bold mb-1">Step {s.step}</div>
                <div className="text-white text-sm font-medium">{s.label}</div>
                <div className="text-gray-500 text-xs mt-0.5">{s.desc}</div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}
