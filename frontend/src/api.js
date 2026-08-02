import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
})

// ── auth token ───────────────────────────────────────────────────────────────
const TOKEN_KEY = 'recomai_token'

export const getToken   = () => localStorage.getItem(TOKEN_KEY)
export const setToken   = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

// Attach the bearer token to every request. A header (rather than a cookie)
// keeps this working across origins — the SPA and the API are on different
// domains in production.
api.interceptors.request.use(config => {
  const token = getToken()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// A 401 means the token is missing, expired, or was signed with a secret the
// server no longer has. Drop it so the app falls back to the login screen
// instead of retrying with a credential that cannot work.
api.interceptors.response.use(
  r => r,
  err => {
    if (err.response?.status === 401) clearToken()
    return Promise.reject(err)
  },
)

// ── recommendations ──────────────────────────────────────────────────────────
export const fetchRecommendations = (userId, n = 10) =>
  api.get('/recommend', { params: { user_id: userId, n } }).then(r => r.data)

export const logClick = (userId, movieIdx) =>
  api.post('/log_click', { user_id: userId, movie_idx: movieIdx })

export const fetchUsers = (n = 30) =>
  api.get('/users', { params: { n } }).then(r => r.data)

export const fetchGenres = () =>
  api.get('/genres').then(r => r.data)

export const registerUser = (name, genres) =>
  api.post('/users/register', { name, genres }).then(r => r.data)

export const fetchHealth = () =>
  api.get('/health').then(r => r.data)

// ── auth ─────────────────────────────────────────────────────────────────────
export const authRegister = (username, password, displayName, genres = []) =>
  api.post('/auth/register', {
    username, password, display_name: displayName, genres,
  }).then(r => r.data)

export const authLogin = (username, password) =>
  api.post('/auth/login', { username, password }).then(r => r.data)

export const authMe = () =>
  api.get('/auth/me').then(r => r.data)

export const saveGenres = (genres) =>
  api.put('/auth/genres', { genres }).then(r => r.data)

// ── my list ──────────────────────────────────────────────────────────────────
export const fetchMyList = (limit = 15) =>
  api.get('/my-list', { params: { limit } }).then(r => r.data)

export const saveToMyList = (movie) =>
  api.post('/my-list', {
    movie_idx : movie.movie_idx,
    title     : movie.title,
    genres    : movie.genres ?? '',
    poster_url: movie.poster_url ?? null,
    avg_rating: movie.avg_rating ?? null,
  }).then(r => r.data)

export const removeFromMyList = (movieIdx) =>
  api.delete(`/my-list/${movieIdx}`).then(r => r.data)

export default api
