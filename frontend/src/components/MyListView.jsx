import { motion } from 'framer-motion'
import { MY_LIST_SIZE } from '../hooks/useMyList'

/**
 * Grid of saved titles.
 *
 * Each row carries its own poster/title because the engine stops recommending
 * a movie once it's been opened — the recommendation feed can't be used as the
 * source of truth for what the user already picked.
 */
export default function MyListView({ items, loading, onRemove, onOpen, onBack }) {
  return (
    <div className="px-6 md:px-12 pt-24 pb-16 min-h-screen">
      <div className="flex items-end justify-between mb-6">
        <div>
          <h1 className="text-white font-bold text-2xl md:text-3xl">My List</h1>
          <p className="text-[#737373] text-sm mt-1">
            {loading
              ? 'Loading…'
              : items.length === 0
                ? 'Titles you open are kept here.'
                : `${items.length} of your ${MY_LIST_SIZE} most recent picks`}
          </p>
        </div>
        <button
          onClick={onBack}
          className="text-sm text-[#b3b3b3] hover:text-white transition-colors"
        >
          ← Back to browse
        </button>
      </div>

      {/* ── Empty ─────────────────────────────────────────────────────────── */}
      {!loading && items.length === 0 && (
        <div className="rounded-lg py-20 text-center"
             style={{ background: '#181818', border: '1px solid #222' }}>
          <p className="text-white font-semibold mb-1">Nothing saved yet</p>
          <p className="text-[#737373] text-sm max-w-md mx-auto">
            Open any title from the browse page and it will show up here — handy,
            since opening one removes it from your recommendations.
          </p>
        </div>
      )}

      {/* ── Skeletons ─────────────────────────────────────────────────────── */}
      {loading && (
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="rounded-md animate-pulse"
                 style={{ aspectRatio: '2/3', background: '#2f2f2f' }} />
          ))}
        </div>
      )}

      {/* ── Grid ──────────────────────────────────────────────────────────── */}
      {!loading && items.length > 0 && (
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3">
          {items.map((m, i) => (
            <motion.div
              key={m.movie_idx}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.03, 0.3) }}
              className="group relative"
            >
              <div
                onClick={() => onOpen?.(m)}
                className="relative rounded-md overflow-hidden cursor-pointer"
                style={{ aspectRatio: '2/3', background: '#2f2f2f' }}
              >
                {m.poster_url ? (
                  <img src={m.poster_url} alt={m.title}
                       className="absolute inset-0 w-full h-full object-cover"
                       loading="lazy" />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-4xl font-black"
                       style={{ color: 'rgba(255,255,255,0.08)' }}>
                    {m.title?.[0]?.toUpperCase() ?? '?'}
                  </div>
                )}

                <div className="absolute inset-x-0 bottom-0 px-2 pb-2 pt-8"
                     style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.95) 40%, transparent)' }}>
                  <p className="text-white text-[11px] font-bold leading-tight line-clamp-2">
                    {m.title?.replace(/\s*\(\d{4}\)\s*$/, '')}
                  </p>
                </div>
              </div>

              <button
                onClick={(e) => { e.stopPropagation(); onRemove(m.movie_idx) }}
                aria-label={`Remove ${m.title} from My List`}
                className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/80 hover:bg-[#E50914]
                           flex items-center justify-center opacity-0 group-hover:opacity-100
                           focus:opacity-100 transition-opacity"
              >
                <svg className="w-3.5 h-3.5 text-white" viewBox="0 0 24 24" fill="none">
                  <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2.5"
                        strokeLinecap="round" />
                </svg>
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}
