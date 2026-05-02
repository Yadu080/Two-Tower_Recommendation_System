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
const RANK_BADGE = ['', '🥇', '🥈', '🥉']

export default function MovieCard({ movie, onClick, showDebug }) {
  const [hovered, setHovered] = useState(false)

  const primaryGenre = movie.genres?.split('|')[0] ?? ''
  const allGenres    = movie.genres?.split('|') ?? []
  const colors       = GENRE_COLORS[primaryGenre] ?? DEFAULT_COLORS
  const initial      = movie.title?.[0]?.toUpperCase() ?? '?'
  const year         = movie.title?.match(/\((\d{4})\)/)?.[1] ?? ''
  const titleClean   = movie.title?.replace(/\s*\(\d{4}\)\s*$/, '') ?? movie.title

  return (
    <motion.div
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      onClick={() => onClick(movie)}
      whileHover={{ scale: 1.04, zIndex: 10 }}
      transition={{ type: 'spring', stiffness: 300, damping: 22 }}
      className="relative rounded-md overflow-hidden cursor-pointer select-none"
      style={{ aspectRatio: '2/3' }}
    >
      {/* Poster background */}
      <div className="absolute inset-0"
           style={{ background: `linear-gradient(160deg, ${colors.c1} 0%, ${colors.c2} 100%)` }} />

      {/* Big decorative initial */}
      <div className="absolute inset-0 flex items-center justify-center text-[7rem] font-black leading-none pointer-events-none"
           style={{ color: 'rgba(255,255,255,0.06)' }}>
        {initial}
      </div>

      {/* Rank badge top-3 */}
      {movie.rank <= 3 && (
        <div className="absolute top-2 left-2 text-xl leading-none z-10">
          {RANK_BADGE[movie.rank]}
        </div>
      )}

      {/* Bottom gradient info */}
      <div className="absolute inset-x-0 bottom-0 z-10"
           style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.95) 60%, transparent 100%)' }}>
        <div className="px-3 pb-3 pt-8">
          <h3 className="text-white font-bold text-sm leading-tight line-clamp-2 mb-1">{titleClean}</h3>
          <div className="flex items-center gap-2 mb-1.5">
            {year && <span className="text-[#46d369] text-xs font-semibold">{year}</span>}
            <span className="text-[#b3b3b3] text-xs">★ {movie.avg_rating?.toFixed(1)}</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {allGenres.slice(0, 2).map(g => (
              <span key={g} className="text-[10px] px-1.5 py-0.5 rounded-sm bg-white/10 text-white/80">{g}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Hover overlay */}
      <AnimatePresence>
        {hovered && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="absolute inset-0 z-20 flex flex-col justify-end"
            style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.98) 50%, rgba(0,0,0,0.6) 100%)' }}
          >
            <div className="p-3 space-y-2">
              <p className="text-white font-bold text-sm leading-tight">{titleClean}</p>
              <p className="text-[#E50914] text-xs font-semibold">✦ {movie.why_recommended}</p>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-[10px] text-[#737373]">
                  <span>Match score</span>
                  <span className="text-[#46d369] font-semibold">{Math.round(movie.ranking_score * 100)}%</span>
                </div>
                <div className="h-1 bg-[#333] rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${movie.ranking_score * 100}%` }}
                    transition={{ duration: 0.4, ease: 'easeOut' }}
                    className="h-full bg-[#46d369] rounded-full"
                  />
                </div>
              </div>
              {showDebug && (
                <div className="text-[10px] text-[#555] space-y-0.5 border-t border-[#333] pt-1.5">
                  <div className="flex justify-between"><span>Embed sim</span><span>{movie.embedding_sim?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Rank score</span><span>{movie.ranking_score?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Popularity</span><span>{movie.popularity?.toFixed(3)}</span></div>
                  <div className="flex justify-between"><span>Latency</span><span>{movie.latency_ms}ms</span></div>
                </div>
              )}
              <div className="flex flex-wrap gap-1 pt-0.5">
                {allGenres.map(g => (
                  <span key={g} className="text-[10px] px-1.5 py-0.5 rounded-sm bg-white/10 text-white/70">{g}</span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
