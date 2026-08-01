import { useRef, useState, useEffect, useCallback } from 'react'
import MovieCard from './MovieCard'

/**
 * Horizontal carousel of posters with hover-arrows, in the style of a
 * streaming-service "row". Scrolls by one viewport-width per arrow click.
 */
export default function Row({ title, movies, onCardClick, showDebug, numbered = false }) {
  const trackRef = useRef(null)
  const [canLeft,  setCanLeft]  = useState(false)
  const [canRight, setCanRight] = useState(false)

  const updateArrows = useCallback(() => {
    const el = trackRef.current
    if (!el) return
    // 2px slack so sub-pixel rounding doesn't leave a permanently-enabled arrow
    setCanLeft(el.scrollLeft > 2)
    setCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2)
  }, [])

  useEffect(() => {
    updateArrows()
    const el = trackRef.current
    if (!el) return
    // recompute when the track resizes (window resize, poster images loading)
    const ro = new ResizeObserver(updateArrows)
    ro.observe(el)
    return () => ro.disconnect()
  }, [updateArrows, movies])

  const scrollBy = (dir) => {
    const el = trackRef.current
    if (!el) return
    el.scrollBy({ left: dir * el.clientWidth * 0.9, behavior: 'smooth' })
  }

  if (!movies?.length) return null

  return (
    <section className="group/row relative mb-8 md:mb-12">
      <h2 className="text-white font-bold text-lg md:text-xl mb-2 px-6 md:px-12">
        {title}
      </h2>

      <div className="relative">
        {/* ── Left arrow ────────────────────────────────────────────────── */}
        {canLeft && (
          <button
            onClick={() => scrollBy(-1)}
            aria-label="Scroll left"
            className="absolute left-0 top-0 bottom-0 z-30 w-6 md:w-12 flex items-center justify-center
                       bg-black/40 hover:bg-black/70 opacity-0 group-hover/row:opacity-100 transition-opacity"
          >
            <svg className="w-5 h-5 md:w-7 md:h-7 text-white" viewBox="0 0 24 24" fill="none">
              <path d="M15 5l-7 7 7 7" stroke="currentColor" strokeWidth="2.5"
                    strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}

        {/* ── Track ─────────────────────────────────────────────────────── */}
        <div
          ref={trackRef}
          onScroll={updateArrows}
          className="nf-row-track px-6 md:px-12 py-4"
        >
          {movies.map((movie, i) => (
            <div
              key={movie.movie_idx}
              className={`flex-none ${
                numbered
                  ? 'flex items-end w-[46vw] sm:w-[34vw] md:w-[25vw] lg:w-[19vw] xl:w-[15vw]'
                  : 'w-[31vw] sm:w-[23vw] md:w-[18vw] lg:w-[14.5vw] xl:w-[12vw]'
              }`}
            >
              {numbered && (
                <span
                  className="nf-rank-numeral select-none pointer-events-none
                             text-[5.5rem] md:text-[7rem] lg:text-[8rem] -mr-3 md:-mr-5"
                >
                  {i + 1}
                </span>
              )}
              <div className={numbered ? 'flex-1 min-w-0' : 'w-full'}>
                <MovieCard movie={movie} onClick={onCardClick} showDebug={showDebug} />
              </div>
            </div>
          ))}
        </div>

        {/* ── Right arrow ───────────────────────────────────────────────── */}
        {canRight && (
          <button
            onClick={() => scrollBy(1)}
            aria-label="Scroll right"
            className="absolute right-0 top-0 bottom-0 z-30 w-6 md:w-12 flex items-center justify-center
                       bg-black/40 hover:bg-black/70 opacity-0 group-hover/row:opacity-100 transition-opacity"
          >
            <svg className="w-5 h-5 md:w-7 md:h-7 text-white" viewBox="0 0 24 24" fill="none">
              <path d="M9 5l7 7-7 7" stroke="currentColor" strokeWidth="2.5"
                    strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
    </section>
  )
}
