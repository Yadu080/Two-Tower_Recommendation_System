import { useState, useCallback, useMemo, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import LandingPage  from './components/LandingPage'
import AuthPage     from './components/AuthPage'
import GenrePicker  from './components/GenrePicker'
import Navbar       from './components/Navbar'
import Billboard    from './components/Billboard'
import Row          from './components/Row'
import DetailModal  from './components/DetailModal'
import DemoDrawer   from './components/DemoDrawer'
import MyListView   from './components/MyListView'
import useMyList    from './hooks/useMyList'
import {
  fetchRecommendations, registerUser, logClick,
  authLogin, authRegister, authMe, saveGenres,
  getToken, setToken, clearToken,
} from './api'

// ── view state machine ─────────────────────────────────────────────────────
//  landing → auth → genres → recs ⇄ mylist

const REC_COUNT = 24   // enough titles to fill a billboard plus several rows

/**
 * Slice a flat ranked list into themed carousels.
 * The API returns one ranked list, so rows are derived client-side: the head
 * becomes the Top 10, and the tail is grouped by the user's strongest genres.
 */
function buildRows(recs, userGenres) {
  if (!recs.length) return []

  const rows = []
  const top10 = recs.slice(0, 10)
  rows.push({ key: 'top10', title: 'Top 10 for You Today', movies: top10, numbered: true })

  // Group the remaining titles under whichever of the user's genres they match.
  const rest = recs.slice(10)
  const used = new Set()

  for (const genre of userGenres ?? []) {
    const matches = rest.filter(m => !used.has(m.movie_idx) && m.genres?.split('|').includes(genre))
    if (matches.length >= 3) {
      matches.forEach(m => used.add(m.movie_idx))
      rows.push({ key: `genre-${genre}`, title: `Because you like ${genre}`, movies: matches })
    }
  }

  // Anything left over, plus a couple of alternate cuts over the full list.
  const leftover = rest.filter(m => !used.has(m.movie_idx))
  if (leftover.length >= 3) {
    rows.push({ key: 'more', title: 'More Like This', movies: leftover })
  }

  const trending = [...recs].sort((a, b) => (b.popularity ?? 0) - (a.popularity ?? 0)).slice(0, 12)
  rows.push({ key: 'trending', title: 'Trending Now', movies: trending })

  const acclaimed = [...recs].sort((a, b) => (b.avg_rating ?? 0) - (a.avg_rating ?? 0)).slice(0, 12)
  rows.push({ key: 'acclaimed', title: 'Critically Acclaimed', movies: acclaimed })

  return rows
}

export default function App() {
  const [view, setView]                 = useState('landing')
  const [authMode, setAuthMode]         = useState('login')
  const [pendingName, setPendingName]   = useState('')
  const [account, setAccount]           = useState(null)   // signed-in account
  const [currentUser, setCurrentUser]   = useState(null)   // {id,name,genres,is_new}
  const [recs, setRecs]                 = useState([])
  const [loading, setLoading]           = useState(false)
  const [error, setError]               = useState(null)
  const [showDebug, setShowDebug]       = useState(false)
  const [showDemo, setShowDemo]         = useState(false)
  const [detailMovie, setDetailMovie]   = useState(null)
  const [clickedItems, setClickedItems] = useState(new Set())
  const [bootstrapping, setBootstrapping] = useState(!!getToken())

  const isAuthed = !!account
  const myList = useMyList(isAuthed)

  // ── load recommendations for a given user id ────────────────────────────
  const loadRecs = useCallback(async (userId) => {
    setLoading(true)
    setError(null)
    setRecs([])
    try {
      const data = await fetchRecommendations(userId, REC_COUNT)
      setRecs(data.results)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Could not fetch recommendations. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  // ── restore an existing session on first load ───────────────────────────
  useEffect(() => {
    if (!getToken()) return
    let cancelled = false

    authMe()
      .then(profile => {
        if (cancelled) return
        setAccount(profile)
        // an account with no genres yet still needs to pick them
        if (!profile.genres?.length || profile.rec_user_id == null) {
          setPendingName(profile.name)
          setView('genres')
        } else {
          const user = {
            id: profile.rec_user_id, name: profile.name,
            genres: profile.genres, is_new: true,
          }
          setCurrentUser(user)
          setView('recs')
          loadRecs(user.id)
        }
      })
      .catch(() => { clearToken() })   // expired or invalid — fall back to landing
      .finally(() => { if (!cancelled) setBootstrapping(false) })

    return () => { cancelled = true }
  }, [loadRecs])

  // ── auth ────────────────────────────────────────────────────────────────
  const applyAuth = (data) => {
    setToken(data.access_token)
    const profile = data.user
    setAccount(profile)

    if (!profile.genres?.length || profile.rec_user_id == null) {
      setPendingName(profile.name)
      setView('genres')
      return
    }
    const user = {
      id: profile.rec_user_id, name: profile.name,
      genres: profile.genres, is_new: true,
    }
    setCurrentUser(user)
    setClickedItems(new Set())
    setView('recs')
    loadRecs(user.id)
  }

  const handleLogin  = async (username, password) =>
    applyAuth(await authLogin(username, password))

  const handleSignup = async (username, password) =>
    applyAuth(await authRegister(username, password, username, []))

  const handleSignOut = () => {
    clearToken()
    setAccount(null)
    setCurrentUser(null)
    setRecs([])
    setPendingName('')
    setView('landing')
  }

  // ── Step 1: name entered on landing page (guest flow) ───────────────────
  const handleNameSubmit = (name) => {
    setPendingName(name)
    setView('genres')
  }

  // ── Step 2: genres chosen ─────────────────────────────────────────────
  const handleGenresSubmit = async (genres) => {
    // errors propagate to GenrePicker, which shows them and resets its loading state
    if (isAuthed) {
      const profile = await saveGenres(genres)
      setAccount(profile)
      const user = {
        id: profile.rec_user_id, name: profile.name,
        genres: profile.genres, is_new: true,
      }
      setCurrentUser(user)
      setClickedItems(new Set())
      setView('recs')
      loadRecs(user.id)
      return
    }

    const data = await registerUser(pendingName, genres)
    const user = { id: data.user_id, name: data.name, genres: data.genres, is_new: true }
    setCurrentUser(user)
    setClickedItems(new Set())
    setView('recs')
    loadRecs(user.id)
  }

  // ── Demo profile selected ──────────────────────────────────────────────
  const handleDemoSelect = (user) => {
    setCurrentUser(user)
    setClickedItems(new Set())
    setShowDemo(false)
    setView('recs')
    loadRecs(user.id)
  }

  // ── Card clicked (save + log + refresh) ────────────────────────────────
  const handleCardClick = async (movie) => {
    setDetailMovie(movie)
    // Save before anything else: the engine drops seen titles from future
    // responses, so this is the last chance to capture it.
    myList.add(movie)

    if (clickedItems.has(movie.movie_idx)) return
    setClickedItems(prev => new Set([...prev, movie.movie_idx]))
    await logClick(currentUser.id, movie.movie_idx).catch(() => {})
    loadRecs(currentUser.id)
  }

  // "Play" is a demo affordance — treat it as a strong implicit signal.
  const handlePlay = async (movie) => {
    setDetailMovie(null)
    myList.add(movie)

    if (clickedItems.has(movie.movie_idx)) return
    setClickedItems(prev => new Set([...prev, movie.movie_idx]))
    await logClick(currentUser.id, movie.movie_idx).catch(() => {})
    loadRecs(currentUser.id)
  }

  const rows = useMemo(
    () => buildRows(recs, currentUser?.genres),
    [recs, currentUser?.genres],
  )

  // ── restoring a session: hold the landing page back briefly ─────────────
  if (bootstrapping) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#141414' }}>
        <div className="w-10 h-10 rounded bg-[#E50914] flex items-center justify-center font-bold text-white animate-pulse">
          R
        </div>
      </div>
    )
  }

  const showChrome = view === 'recs' || view === 'mylist'

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

      <AnimatePresence>
        {detailMovie && (
          <DetailModal
            movie={detailMovie}
            onClose={() => setDetailMovie(null)}
            onPlay={handlePlay}
          />
        )}
      </AnimatePresence>

      {showChrome && (
        <Navbar
          user={account ?? currentUser}
          view={view}
          isAuthed={isAuthed}
          myListCount={myList.items.length}
          onNavigate={setView}
          onSwitchProfile={isAuthed ? handleSignOut : () => setView('landing')}
          onSignIn={() => { setAuthMode('login'); setView('auth') }}
          onDemoClick={() => setShowDemo(true)}
          showDebug={showDebug}
          onToggleDebug={() => setShowDebug(v => !v)}
        />
      )}

      <AnimatePresence mode="wait">
        {/* ── LANDING ─────────────────────────────────────────────────── */}
        {view === 'landing' && (
          <motion.div key="landing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <LandingPage
              onNameSubmit={handleNameSubmit}
              onDemoClick={() => setShowDemo(true)}
              onSignIn={() => { setAuthMode('login'); setView('auth') }}
              onSignUp={() => { setAuthMode('signup'); setView('auth') }}
            />
          </motion.div>
        )}

        {/* ── AUTH ────────────────────────────────────────────────────── */}
        {view === 'auth' && (
          <motion.div key="auth" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <AuthPage
              mode={authMode}
              onSubmit={authMode === 'signup' ? handleSignup : handleLogin}
              onSwitchMode={() => setAuthMode(m => (m === 'signup' ? 'login' : 'signup'))}
              onBack={() => setView('landing')}
            />
          </motion.div>
        )}

        {/* ── GENRE PICKER ────────────────────────────────────────────── */}
        {view === 'genres' && (
          <motion.div key="genres" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -40 }}>
            <GenrePicker
              name={pendingName || account?.name}
              initialSelected={account?.genres ?? []}
              onComplete={handleGenresSubmit}
              onBack={() => setView(currentUser ? 'recs' : 'landing')}
            />
          </motion.div>
        )}

        {/* ── MY LIST ─────────────────────────────────────────────────── */}
        {view === 'mylist' && (
          <motion.div key="mylist" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="min-h-screen" style={{ background: '#141414' }}>
            <MyListView
              items={myList.items}
              loading={myList.loading}
              onRemove={myList.remove}
              onOpen={setDetailMovie}
              onBack={() => setView('recs')}
            />
          </motion.div>
        )}

        {/* ── BROWSE ──────────────────────────────────────────────────── */}
        {view === 'recs' && (
          <motion.div key="recs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="min-h-screen" style={{ background: '#141414' }}>

            {/* ── Error ───────────────────────────────────────────────── */}
            {error && (
              <div className="pt-24 px-6 md:px-12">
                <div className="p-4 rounded-lg text-sm text-red-300"
                     style={{ background: 'rgba(127,29,29,0.3)', border: '1px solid rgba(153,27,27,0.5)' }}>
                  {error}
                </div>
              </div>
            )}

            {/* ── Loading skeleton ────────────────────────────────────── */}
            {loading && !error && (
              <div className="pt-16">
                <div className="w-full h-[48vw] max-h-[68vh] min-h-[440px] animate-pulse"
                     style={{ background: 'linear-gradient(to top, #141414 0%, #222 60%)' }} />
                <div className="px-6 md:px-12 -mt-20 relative z-10 space-y-8">
                  {[0, 1].map(r => (
                    <div key={r}>
                      <div className="h-5 w-48 rounded mb-3 animate-pulse" style={{ background: '#2f2f2f' }} />
                      <div className="flex gap-1">
                        {Array.from({ length: 7 }).map((_, i) => (
                          <div key={i} className="flex-none w-[31vw] sm:w-[23vw] md:w-[18vw] lg:w-[14.5vw] xl:w-[12vw]
                                                  rounded-md animate-pulse"
                               style={{ aspectRatio: '2/3', background: '#2f2f2f' }} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Billboard + rows ────────────────────────────────────── */}
            {!loading && !error && recs.length > 0 && (
              <>
                <Billboard
                  movie={recs[0]}
                  onPlay={handlePlay}
                  onMoreInfo={setDetailMovie}
                />

                {/* Rows ride up into the billboard's lower fade. Keep this pull
                    smaller than the billboard's bottom inset or the first row
                    heading collides with the Play / More Info buttons. */}
                <div className="relative z-10 -mt-[6vw] pb-16">
                  {rows.map(row => (
                    <Row
                      key={row.key}
                      title={row.title}
                      movies={row.movies}
                      numbered={row.numbered}
                      onCardClick={handleCardClick}
                      showDebug={showDebug}
                    />
                  ))}

                  {/* Footer */}
                  <footer className="px-6 md:px-12 pt-8 border-t" style={{ borderColor: '#222' }}>
                    {clickedItems.size > 0 && (
                      <p className="text-[#737373] text-xs mb-4">
                        {clickedItems.size} title{clickedItems.size > 1 ? 's' : ''} watched ·
                        recommendations updated live ·{' '}
                        <button onClick={() => setView('mylist')} className="underline hover:text-white">
                          saved to My List
                        </button>
                      </p>
                    )}
                    <div className="flex flex-wrap gap-2 mb-4">
                      {['UserTower → 128-dim embedding', 'ANN top-500 retrieval',
                        'GBM re-rank', 'O(N log K) heap top-24'].map(s => (
                        <span key={s} className="text-xs px-3 py-1.5 rounded-full text-[#737373]"
                              style={{ background: '#1f1f1f', border: '1px solid #2f2f2f' }}>
                          {s}
                        </span>
                      ))}
                    </div>
                    <p className="text-[#555] text-xs">
                      Two-Tower Neural Retrieval · Trained on MovieLens 20M
                    </p>
                  </footer>
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
