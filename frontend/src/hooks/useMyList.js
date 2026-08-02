import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchMyList, saveToMyList, removeFromMyList } from '../api'

const STORAGE_KEY = 'recomai_my_list'
export const MY_LIST_SIZE = 15

/**
 * "My List" state, backed by the API when signed in and by localStorage when
 * not.
 *
 * The guest path exists because opening a title removes it from the feed —
 * the engine excludes seen items — so without somewhere to keep them, a click
 * would simply lose the movie. localStorage gives that a home before an
 * account does, and the entries are merged into the account on first sign-in.
 */
export default function useMyList(isAuthed) {
  const [items, setItems]   = useState([])
  const [loading, setLoading] = useState(false)
  const mergedRef = useRef(false)

  // ── local storage helpers ─────────────────────────────────────────────────
  const readLocal = () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      return raw ? JSON.parse(raw) : []
    } catch {
      return []   // corrupt or unavailable storage shouldn't break the page
    }
  }

  const writeLocal = (next) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      /* quota or private mode — the in-memory copy still works this session */
    }
  }

  // ── load ──────────────────────────────────────────────────────────────────
  const refresh = useCallback(async () => {
    if (!isAuthed) {
      setItems(readLocal())
      return
    }
    setLoading(true)
    try {
      const data = await fetchMyList(MY_LIST_SIZE)
      setItems(data.items ?? [])
    } catch {
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [isAuthed])

  // On first sign-in, push anything saved as a guest up to the account so the
  // list doesn't appear to reset at the moment of logging in.
  useEffect(() => {
    let cancelled = false

    const run = async () => {
      if (isAuthed && !mergedRef.current) {
        mergedRef.current = true
        const local = readLocal()
        if (local.length) {
          // oldest first, so the newest ends up on top server-side
          for (const m of [...local].reverse()) {
            await saveToMyList(m).catch(() => {})
          }
          writeLocal([])
        }
      }
      if (!cancelled) await refresh()
    }

    run()
    return () => { cancelled = true }
  }, [isAuthed, refresh])

  useEffect(() => {
    if (!isAuthed) mergedRef.current = false
  }, [isAuthed])

  // ── mutations ─────────────────────────────────────────────────────────────
  const add = useCallback(async (movie) => {
    const entry = {
      movie_idx : movie.movie_idx,
      title     : movie.title,
      genres    : movie.genres ?? '',
      poster_url: movie.poster_url ?? null,
      avg_rating: movie.avg_rating ?? null,
      saved_at  : new Date().toISOString(),
    }

    // optimistic: move to front, drop any existing copy, then cap
    setItems(prev => {
      const next = [entry, ...prev.filter(i => i.movie_idx !== entry.movie_idx)]
                     .slice(0, MY_LIST_SIZE)
      if (!isAuthed) writeLocal(next)
      return next
    })

    if (isAuthed) {
      await saveToMyList(entry).catch(() => {})
    }
  }, [isAuthed])

  const remove = useCallback(async (movieIdx) => {
    setItems(prev => {
      const next = prev.filter(i => i.movie_idx !== movieIdx)
      if (!isAuthed) writeLocal(next)
      return next
    })
    if (isAuthed) {
      await removeFromMyList(movieIdx).catch(() => {})
    }
  }, [isAuthed])

  return { items, loading, add, remove, refresh }
}
