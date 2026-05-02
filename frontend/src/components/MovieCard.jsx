import { motion } from 'framer-motion'

const GENRE_COLORS = {
  Action: 'bg-red-500/20 text-red-300',
  Comedy: 'bg-yellow-500/20 text-yellow-300',
  Drama: 'bg-blue-500/20 text-blue-300',
  Thriller: 'bg-purple-500/20 text-purple-300',
  Romance: 'bg-pink-500/20 text-pink-300',
  'Sci-Fi': 'bg-cyan-500/20 text-cyan-300',
  Horror: 'bg-orange-500/20 text-orange-300',
  Animation: 'bg-green-500/20 text-green-300',
  Adventure: 'bg-emerald-500/20 text-emerald-300',
  Documentary: 'bg-gray-500/20 text-gray-300',
}

function ScoreBar({ value, color = 'bg-purple-500' }) {
  return (
    <div className="w-full bg-gray-700 rounded-full h-1.5">
      <motion.div
        className={`h-1.5 rounded-full ${color}`}
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(value * 100, 100)}%` }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      />
    </div>
  )
}

export default function MovieCard({ movie, onClick, showDebug }) {
  const genres = movie.genres ? movie.genres.split('|').slice(0, 3) : []
  const year = movie.title?.match(/\((\d{4})\)/)?.[1]
  const cleanTitle = movie.title?.replace(/\s*\(\d{4}\)/, '') ?? 'Unknown'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ duration: 0.2 }}
      onClick={() => onClick(movie)}
      className="bg-gray-800 border border-gray-700 rounded-xl p-4 cursor-pointer
                 hover:border-purple-500/50 hover:bg-gray-750 transition-colors
                 flex flex-col gap-3 group"
    >
      {/* Rank badge + title */}
      <div className="flex items-start gap-3">
        <span className="text-2xl font-bold text-gray-600 group-hover:text-purple-400
                         transition-colors min-w-[2rem] text-center">
          {movie.rank}
        </span>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-white text-sm leading-tight truncate">
            {cleanTitle}
          </h3>
          {year && <span className="text-xs text-gray-500">{year}</span>}
        </div>
        {/* Star rating */}
        <div className="flex items-center gap-1 shrink-0">
          <span className="text-yellow-400 text-xs">★</span>
          <span className="text-gray-300 text-xs">
            {movie.avg_rating?.toFixed(1) ?? '—'}
          </span>
        </div>
      </div>

      {/* Genres */}
      <div className="flex flex-wrap gap-1">
        {genres.map(g => (
          <span
            key={g}
            className={`text-xs px-2 py-0.5 rounded-full ${
              GENRE_COLORS[g] ?? 'bg-gray-700 text-gray-400'
            }`}
          >
            {g}
          </span>
        ))}
      </div>

      {/* Why recommended */}
      <p className="text-xs text-purple-300 italic truncate">
        ✦ {movie.why_recommended}
      </p>

      {/* Debug panel */}
      {showDebug && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="border-t border-gray-700 pt-2 space-y-2"
        >
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Embedding similarity</span>
              <span>{movie.embedding_sim?.toFixed(3)}</span>
            </div>
            <ScoreBar value={movie.embedding_sim ?? 0} color="bg-blue-500" />
          </div>
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Ranking score</span>
              <span>{movie.ranking_score?.toFixed(3)}</span>
            </div>
            <ScoreBar value={movie.ranking_score ?? 0} color="bg-purple-500" />
          </div>
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>Popularity</span>
              <span>{movie.popularity?.toFixed(3)}</span>
            </div>
            <ScoreBar value={movie.popularity ?? 0} color="bg-green-500" />
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
