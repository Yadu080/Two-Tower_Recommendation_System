import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

export const fetchRecommendations = (userId, n = 10) =>
  api.get('/recommend', { params: { user_id: userId, n } }).then(r => r.data)

export const logClick = (userId, movieIdx) =>
  api.post('/log_click', { user_id: userId, movie_idx: movieIdx })

export const fetchUsers = (n = 30) =>
  api.get('/users', { params: { n } }).then(r => r.data)

export const fetchHealth = () =>
  api.get('/health').then(r => r.data)
