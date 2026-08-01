import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

/**
 * Expanded detail view for a single title, opened from the billboard or a card.
 * Shows the model's reasoning alongside the usual metadata.
 */
export default function DetailModal({ movie, onClose, onPlay }) {
  const [imgError, setImgError] = useState(false)

  // close on Escape, and prevent the page behind from scrolling
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  if (!movie) return null

  const year       = movie.title?.match(/\((\d{4})\)/)?.[1] ?? ''
  const titleClean = movie.title?.replace(/\s*\(\d{4}\)\s*$/, '') ?? movie.title
  const genres     = movie.genres?.split('|') ?? []
  const hasImage   = movie.poster_url && !imgError
  const match      = Math.round((movie.ranking_score ?? 0) * 100)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      className="fixed inset-0 z-[60] bg-black/80 overflow-y-auto py-8 px-4"
    >
      <motion.div
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 260, damping: 26 }}
        onClick={(e) => e.stopPropagation()}
        className="relative mx-auto max-w-3xl rounded-lg overflow-hidden shadow-2xl"
        style={{ background: '#181818' }}
      >
        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <div className="relative h-[42vh] min-h-[260px]">
          {hasImage ? (
            <img
              src={movie.poster_url}
              alt=""
              onError={() => setImgError(true)}
              className="absolute inset-0 w-full h-full object-cover object-[50%_25%]"
            />
          ) : (
            <div className="absolute inset-0"
                 style={{ background: 'linear-gradient(135deg, #3f1d1d 0%, #181818 70%)' }} />
          )}
          <div className="absolute inset-0 pointer-events-none"
               style={{ background: 'linear-gradient(to top, #181818 0%, transparent 55%)' }} />

          <button
            onClick={onClose}
            aria-label="Close"
            className="absolute top-4 right-4 w-9 h-9 rounded-full bg-[#181818] hover:bg-[#282828]
                       flex items-center justify-center transition-colors"
          >
            <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none">
              <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>

          <div className="absolute bottom-6 left-6 right-6">
            <h2 className="text-white font-black text-2xl md:text-4xl leading-tight mb-4 drop-shadow-lg">
              {titleClean}
            </h2>
            <button onClick={() => onPlay(movie)} className="nf-play">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 4l14 8-14 8V4z" />
              </svg>
              Play
            </button>
          </div>
        </div>

        {/* ── Body ───────────────────────────────────────────────────────── */}
        <div className="p-6 grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-3">
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <span className="text-[#46d369] font-semibold">{match}% Match</span>
              {year && <span className="text-white/80">{year}</span>}
              <span className="px-1.5 py-0.5 border border-white/40 text-white/70 text-xs rounded-sm">HD</span>
            </div>

            {movie.why_recommended && (
              <p className="text-white text-sm md:text-base leading-relaxed">
                {movie.why_recommended}
              </p>
            )}

            <p className="text-white/60 text-sm">
              Average rating <span className="text-white">★ {movie.avg_rating?.toFixed(1)}</span> across
              all MovieLens viewers.
            </p>
          </div>

          <div className="space-y-3 text-sm">
            <div>
              <span className="text-white/50">Genres: </span>
              <span className="text-white">{genres.join(', ')}</span>
            </div>

            {/* Why the model surfaced this title */}
            <div className="pt-2 border-t" style={{ borderColor: '#333' }}>
              <p className="text-white/50 text-xs mb-2">Model signals</p>
              <div className="space-y-1.5 text-xs">
                <Signal label="Embedding similarity" value={movie.embedding_sim} />
                <Signal label="Ranking score"        value={movie.ranking_score} />
                <Signal label="Popularity prior"     value={movie.popularity} />
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

/** Labelled 0–1 signal rendered as a small progress bar. */
function Signal({ label, value }) {
  if (value == null) return null
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-white/60">
        <span>{label}</span>
        <span className="text-white/80">{value.toFixed(3)}</span>
      </div>
      <div className="h-1 rounded-full overflow-hidden" style={{ background: '#333' }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: '#E50914' }} />
      </div>
    </div>
  )
}
