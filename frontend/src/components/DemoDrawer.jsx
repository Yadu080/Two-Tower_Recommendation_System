import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchUsers } from '../api'

export default function DemoDrawer({ onSelect, onClose }) {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchUsers(30)
      .then(d => { setUsers(d.users); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const existing = users.filter(u => !u.is_new)
  const custom   = users.filter(u => u.is_new)

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-end sm:items-center justify-center"
        style={{ background: 'rgba(0,0,0,0.75)' }}
        onClick={e => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ y: 60, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 60, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 280, damping: 26 }}
          className="w-full sm:max-w-md rounded-t-2xl sm:rounded-2xl overflow-hidden"
          style={{ background: '#1f1f1f', border: '1px solid #2f2f2f' }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-[#2f2f2f]">
            <h2 className="text-white font-bold">Demo Profiles</h2>
            <button onClick={onClose} className="text-[#737373] hover:text-white transition-colors text-xl">×</button>
          </div>

          <div className="p-4 max-h-96 overflow-y-auto space-y-4">
            {loading && (
              <p className="text-center text-[#737373] text-sm py-4">Loading…</p>
            )}

            {/* Custom users created this session */}
            {custom.length > 0 && (
              <div>
                <p className="text-xs text-[#737373] uppercase tracking-wider mb-2">Your Profiles</p>
                <div className="space-y-1">
                  {custom.map(u => (
                    <button
                      key={u.id}
                      onClick={() => onSelect(u)}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-white/5 transition-colors text-left"
                    >
                      <div className="w-9 h-9 rounded-md flex items-center justify-center text-sm font-bold text-white flex-shrink-0"
                           style={{ background: '#E50914' }}>
                        {u.name[0].toUpperCase()}
                      </div>
                      <div className="min-w-0">
                        <p className="text-white text-sm font-medium">{u.name}</p>
                        <p className="text-[#737373] text-xs truncate">{u.genres?.join(' · ')}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* MovieLens sample users */}
            {existing.length > 0 && (
              <div>
                <p className="text-xs text-[#737373] uppercase tracking-wider mb-2">MovieLens Sample Users</p>
                <div className="grid grid-cols-4 gap-1.5">
                  {existing.map(u => (
                    <button
                      key={u.id}
                      onClick={() => onSelect(u)}
                      className="px-2 py-2 rounded-lg text-xs text-[#b3b3b3] hover:bg-white/10 hover:text-white transition-colors border border-[#2f2f2f]"
                    >
                      {u.id}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
