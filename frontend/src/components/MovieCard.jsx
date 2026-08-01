import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const GENRE_COLORS = {
  Action:      { c1: '#7f1d1d', c2: '#b91c1c' },
  Adventure:   { c1: '#7c2d12', c2: '#c2410c' },
  Animation:   { c1: '#713f12', c2: '#ca8a04' },
  Children:    { c1: '#14532d', c2: '#15803d' },
  Comedy:      { c1: '#166534', c2: '#16a34a' },
  Crime:       { c1: '#134e4a', c2: '#0f766e' },
  Documentary: { c1: '#1e3a5f', c2: '#1d4ed8' },
  Drama:       { c1: '#1e1b4b', c2: '#4338ca' },
  Fantasy:     { c1: '#2e1065', c2: '#7c3aed' },
  'Film-Noir': { c1: '#1c1c1c', c2: '#404040' },
  Horror:      { c1: '#450a0a', c2: '#991b1b' },
  IMAX:        { c1: '#0c1445', c2: '#1d4ed8' },
  Musical:     { c1: '#500724', c2: '#be185d' },
  Mystery:     { c1: '#2e1065', c2: '#6d28d9' },
  Romance:     { c1: '#4c0519', c2: '#e11d48' },
  'Sci-Fi':    { c1: '#082f49', c2: '#0284c7' },
  Thriller:    { c1: '#1c1917', c2: '#78350f' },
  War:         { c1: '#1c1917', c2: '#44403c' },
  Western:     { c1: '#431407', c2: '#92400e' },
}
const DEFAULT_COLORS = { c1: '#1f2937', c2: '#374151' }

export default function MovieCard({ movie, onClick, showDebug }) {
  const [hovered, setHovered]   = useState(false)
  const [imgError, setImgError] = useState(false)

  const primaryGenre = movie.genres?.split('|')[0] ?? ''
  const allGenres    = movie.genres?.split('|') ?? []
  const colors       = GENRE_COLORS[primaryGenre] ?? DEFAULT_COLORS
  const initial      = movie.title?.[0]?.toUpperCase() ?? '?'
  const year         = movie.title?.match(/\((\d{4})\)/)?.[1] ?? ''
  const titleClean   = movie.title?.replace(/\s*\(\d{4}\)\s*$/, '') ?? movie.title
  const hasPoster    = movie.poster_url && !imgError
  const match        = Math.round((movie.ranking_score ?? 0) * 100)

  return (
    <motion.div
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      onClick={() => onClick(movie)}
      animate={{ scale: hovered ? 1.28 : 1, zIndex: hovered ? 40 : 1 }}
      transition={{ type: 'spring', stiffness: 320, damping: 26, mass: 0.6 }}
      className="relative cursor-pointer select-none rounded-md"
      style={{ transformOrigin: 'center bottom' }}
    >
      {/* ── Poster ─────────────────────────────────────────────────────────── */}
      <div
        className="relative rounded-md overflow-hidden shadow-lg"
        style={{ aspectRatio: '2/3' }}
      >
        {hasPoster ? (
          <img
            src={movie.poster_url}
            alt={titleClean}
            onError={() => setImgError(true)}
            className="absolute inset-0 w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <>
            <div
              className="absolute inset-0"
              style={{ background: `linear-gradient(160deg, ${colors.c1} 0%, ${colors.c2} 100%)` }}
            />
            <div
              className="absolute inset-0 flex items-center justify-center text-[5rem] font-black leading-none pointer-events-none"
              style={{ color: 'rgba(255,255,255,0.07)' }}
            >
              {initial}
            </div>
            {/* title is otherwise invisible on the fallback tile */}
            <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/90 to-transparent pt-8">
              <p className="text-white text-[11px] font-bold leading-tight line-clamp-2">{titleClean}</p>
            </div>
          </>
        )}
      </div>

      {/* ── Hover detail panel ─────────────────────────────────────────────── */}
      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full rounded-b-md overflow-hidden shadow-2xl"
            style={{ background: '#181818' }}
          >
            <div className="p-2.5 space-y-1.5">
              {/* Control bar */}
              <div className="flex items-center gap-1.5">
                <span className="w-6 h-6 rounded-full bg-white flex items-center justify-center">
                  <svg className="w-3 h-3 text-black" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 4l14 8-14 8V4z" />
                  </svg>
                </span>
                <span className="w-6 h-6 rounded-full border border-white/40 flex items-center justify-center">
                  <svg className="w-3 h-3 text-white" viewBox="0 0 24 24" fill="none">
                    <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                </span>
                <span className="w-6 h-6 rounded-full border border-white/40 flex items-center justify-center">
                  <svg className="w-3 h-3 text-white" viewBox="0 0 24 24" fill="none">
                    <path d="M7 11v9H4v-9h3zm3 9h7.5a2 2 0 001.94-1.5l1.4-5.6A1.6 1.6 0 0019.3 11H15V6.5A2.5 2.5 0 0012.5 4L10 10v10z"
                          stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                  </svg>
                </span>
              </div>

              <p className="text-white font-bold text-[11px] leading-tight line-clamp-1">{titleClean}</p>

              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="text-[#46d369] font-semibold">{match}% Match</span>
                {year && <span className="text-white/70">{year}</span>}
                <span className="text-white/50">★ {movie.avg_rating?.toFixed(1)}</span>
              </div>

              {movie.why_recommended && (
                <p className="text-[#E50914] text-[10px] font-medium leading-snug line-clamp-2">
                  ✦ {movie.why_recommended}
                </p>
              )}

              <p className="text-white/50 text-[9px] leading-snug line-clamp-1">
                {allGenres.join(' · ')}
              </p>

              {showDebug && (
                <div className="text-[9px] space-y-0.5 border-t pt-1.5"
                     style={{ borderColor: '#333', color: '#666' }}>
                  <div className="flex justify-between"><span>Embed sim</span><span>{movie.embedding_sim?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Rank score</span><span>{movie.ranking_score?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Popularity</span><span>{movie.popularity?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Latency</span><span>{movie.latency_ms}ms</span></div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
