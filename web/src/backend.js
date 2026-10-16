// A Vercel build serves static assets; MuJoCo and model inference stay on an
// explicitly selected server. This module never handles credentials.
export function backendEndpoints(configured = '', pageOrigin = globalThis.location?.origin ?? 'http://localhost') {
  const base = new URL(configured.trim() || pageOrigin);
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
    throw new Error('Use an HTTP(S) backend URL without credentials, query parameters or a fragment.');
  }
  const prefix = base.pathname.replace(/\/+$/, '');
  const http = `${base.origin}${prefix}`;
  const ws = new URL(`${http}/ws`);
  ws.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
  return {base: http, api: path => `${http}/api/${path.replace(/^\/+/, '')}`, ws: ws.href};
}

export function readBackendOverride(storage) {
  try { return (storage ?? globalThis.localStorage)?.getItem('robopup.backendUrl') || ''; } catch { return ''; }
}

export const configuredBackend = readBackendOverride() || import.meta.env?.VITE_ARENA_BACKEND_URL || '';
export const backend = backendEndpoints(configuredBackend);

export function readSessionToken(storage) {
  try { return (storage ?? globalThis.sessionStorage)?.getItem('robopup.accessKey') || ''; } catch { return ''; }
}

export function authenticatedFetch(url, options = {}) {
  const headers = new Headers(options.headers);
  const token = readSessionToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(url, {...options, headers});
}
