import { useState } from 'react'
import { motion } from 'framer-motion'

const MIN_PASSWORD = 8

/**
 * Sign-in / sign-up form.
 *
 * Errors from the API are surfaced verbatim where they're useful ("Username
 * already taken") rather than replaced with a generic message, since those are
 * the ones a user can actually act on.
 */
export default function AuthPage({ mode = 'login', onSubmit, onSwitchMode, onBack }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw]     = useState(false)
  const [error, setError]       = useState(null)
  const [loading, setLoading]   = useState(false)

  const isSignup = mode === 'signup'

  const submit = async (e) => {
    e?.preventDefault()
    const u = username.trim()

    if (!u || !password) {
      setError('Enter a username and password.')
      return
    }
    if (isSignup && password.length < MIN_PASSWORD) {
      setError(`Password must be at least ${MIN_PASSWORD} characters.`)
      return
    }

    setLoading(true)
    setError(null)
    try {
      await onSubmit(u, password)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === 'string'
          ? detail
          : err.response?.status === 0 || !err.response
            ? 'Could not reach the server. It may be waking up — try again in a moment.'
            : 'Something went wrong. Please try again.'
      )
      setLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen flex flex-col" style={{ background: '#000' }}>
      {/* backdrop */}
      <div className="absolute inset-0 opacity-30 pointer-events-none"
           style={{ background: 'radial-gradient(ellipse 60% 50% at 50% 0%, #3f1d1d 0%, #000 70%)' }} />

      <header className="relative z-10 px-6 md:px-12 py-5">
        <button onClick={onBack} className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-[#E50914] flex items-center justify-center font-bold text-white text-sm">R</div>
          <span className="text-[#E50914] font-black text-2xl tracking-tight">RECOMAI</span>
        </button>
      </header>

      <main className="relative z-10 flex-1 flex items-center justify-center px-4 pb-16">
        <motion.form
          onSubmit={submit}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-md rounded-lg p-8 md:p-12"
          style={{ background: 'rgba(0,0,0,0.75)', border: '1px solid #222' }}
        >
          <h1 className="text-white font-bold text-3xl mb-6">
            {isSignup ? 'Sign Up' : 'Sign In'}
          </h1>

          {error && (
            <div className="mb-4 px-4 py-3 rounded text-sm text-white"
                 style={{ background: '#e87c03' }}>
              {error}
            </div>
          )}

          <div className="space-y-4">
            <input
              autoFocus
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Username"
              autoComplete="username"
              maxLength={40}
              className="nf-input"
            />

            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Password"
                autoComplete={isSignup ? 'new-password' : 'current-password'}
                maxLength={200}
                className="nf-input pr-16"
              />
              <button
                type="button"
                onClick={() => setShowPw(v => !v)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-[#8c8c8c] hover:text-white"
              >
                {showPw ? 'Hide' : 'Show'}
              </button>
            </div>

            {isSignup && (
              <p className="text-[#8c8c8c] text-xs">
                At least {MIN_PASSWORD} characters. Please don't reuse a password
                from another site — this is a demo project.
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="nf-btn w-full mt-6 disabled:opacity-60 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                {isSignup ? 'Creating account…' : 'Signing in…'}
              </>
            ) : (isSignup ? 'Sign Up' : 'Sign In')}
          </button>

          {loading && (
            <p className="text-[#737373] text-xs text-center mt-3">
              The server sleeps when idle — the first request can take up to a minute.
            </p>
          )}

          <p className="mt-8 text-[#737373] text-sm">
            {isSignup ? 'Already have an account?' : 'New to RECOMAI?'}{' '}
            <button
              type="button"
              onClick={() => { setError(null); onSwitchMode() }}
              className="text-white hover:underline"
            >
              {isSignup ? 'Sign in' : 'Sign up now'}
            </button>
          </p>
        </motion.form>
      </main>
    </div>
  )
}
