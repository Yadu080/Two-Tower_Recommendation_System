import { useState } from 'react'
import { motion } from 'framer-motion'

/**
 * Full-bleed hero "billboard" for the single strongest recommendation —
 * backdrop image, title, match score, synopsis line and primary actions.
 */
export default function Billboard({ movie, onPlay, onMoreInfo }) {
  const [imgError, setImgError] = useState(false)

  if (!movie) return null

  const year       = movie.title?.match(/\((\d{4})\)/)?.[1] ?? ''
  const titleClean = movie.title?.replace(/\s*\(\d{4}\)\s*$/, '') ?? movie.title
  const genres     = movie.genres?.split('|') ?? []
  const hasImage   = movie.poster_url && !imgError
  const match      = Math.round((movie.ranking_score ?? 0) * 100)

  return (
    <div className="relative w-full h-[48vw] max-h-[68vh] min-h-[440px]">
      {/* ── Backdrop ──────────────────────────────────────────────────────── */}
      {hasImage ? (
        <img
          src={movie.poster_url}
          alt=""
          onError={() => setImgError(true)}
          /* posters are portrait, so bias the crop toward the upper third
             where faces/titles usually sit rather than centre-cropping */
          className="absolute inset-0 w-full h-full object-cover object-[50%_28%]"
        />
      ) : (
        <div
          className="absolute inset-0"
          style={{ background: 'linear-gradient(135deg, #3f1d1d 0%, #141414 70%)' }}
        />
      )}

      {/* Scrims: bottom fade into the page, left fade behind the copy */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: 'linear-gradient(to top, #141414 0%, rgba(20,20,20,0.55) 30%, transparent 60%)' }}
      />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: 'linear-gradient(to right, rgba(20,20,20,0.92) 0%, rgba(20,20,20,0.55) 35%, transparent 65%)' }}
      />

      {/* ── Copy ──────────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        /* bottom inset must stay clear of the row overlap App.jsx applies below */
        className="absolute bottom-[22%] left-0 px-6 md:px-12 max-w-2xl"
      >
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[#E50914] font-black text-sm tracking-[0.2em]">TOP PICK</span>
          <span className="text-white/50 text-sm">for you</span>
        </div>

        <h1 className="text-white font-black text-3xl md:text-5xl lg:text-6xl leading-none mb-4 drop-shadow-lg">
          {titleClean}
        </h1>

        <div className="flex flex-wrap items-center gap-3 mb-4 text-sm">
          <span className="text-[#46d369] font-semibold">{match}% Match</span>
          {year && <span className="text-white/80">{year}</span>}
          <span className="text-white/60">★ {movie.avg_rating?.toFixed(1)}</span>
          <span className="px-1.5 py-0.5 border border-white/40 text-white/70 text-xs rounded-sm">HD</span>
        </div>

        {movie.why_recommended && (
          <p className="text-white/90 text-sm md:text-lg leading-relaxed mb-2 drop-shadow max-w-xl">
            {movie.why_recommended}
          </p>
        )}

        <p className="text-white/50 text-xs md:text-sm mb-6">{genres.join(' · ')}</p>

        <div className="flex items-center gap-3">
          <button onClick={() => onPlay(movie)} className="nf-play">
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 4l14 8-14 8V4z" />
            </svg>
            Play
          </button>
          <button onClick={() => onMoreInfo(movie)} className="nf-info">
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
              <path d="M12 11v5M12 7.5v.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            More Info
          </button>
        </div>
      </motion.div>
    </div>
  )
}
