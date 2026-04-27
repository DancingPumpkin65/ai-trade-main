const STORAGE_KEY = "morocco-trading-auth-token";

export function getStoredToken() {
  return window.localStorage.getItem(STORAGE_KEY);
}

export function setStoredToken(token: string) {
  window.localStorage.setItem(STORAGE_KEY, token);
}

export function clearStoredToken() {
  window.localStorage.removeItem(STORAGE_KEY);
}
