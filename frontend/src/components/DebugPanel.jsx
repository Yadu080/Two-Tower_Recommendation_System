import { motion, AnimatePresence } from 'framer-motion'

export default function DebugPanel({ cacheStats, latencyMs, visible }) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-gray-900 border border-gray-700 rounded-xl p-4 font-mono text-xs overflow-hidden"
        >
          <p className="text-purple-400 font-bold mb-3">⚙ System Debug Panel</p>

          <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-gray-400">
            <span>Pipeline latency</span>
            <span className="text-green-400">{latencyMs ? `${latencyMs} ms` : '—'}</span>

            <span>Cache hits</span>
            <span className="text-blue-400">{cacheStats?.hits ?? '—'}</span>

            <span>Cache misses</span>
            <span className="text-yellow-400">{cacheStats?.misses ?? '—'}</span>

            <span>Cache hit rate</span>
            <span className="text-blue-400">
              {cacheStats?.hit_rate != null
                ? `${(cacheStats.hit_rate * 100).toFixed(1)}%`
                : '—'}
            </span>

            <span>Cache size</span>
            <span className="text-gray-300">
              {cacheStats?.size ?? '—'} / {cacheStats?.capacity ?? '—'}
            </span>
          </div>

          <p className="text-gray-600 mt-3">
            Retrieval: FAISS IndexFlatIP → 500 candidates → LightGBM → top-10
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
