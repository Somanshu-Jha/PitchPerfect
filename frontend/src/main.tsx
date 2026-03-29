import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// --- GLOBAL FETCH INTERCEPTOR FOR DEBUGGING ---
const originalFetch = window.fetch;
window.fetch = async (...args) => {
  const [resource] = args;
  console.log(`[Fetch Interceptor] 🚀 Starting request to: ${resource}`);
  try {
    const response = await originalFetch(...args);
    console.log(`[Fetch Interceptor] ✅ Response from: ${resource} | Status: ${response.status}`);
    return response;
  } catch (error) {
    console.error(`[Fetch Interceptor] ❌ Request to ${resource} failed:`, error);
    throw error;
  }
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
