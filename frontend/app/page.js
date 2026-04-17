"use client";

import { useState } from 'react';
import { Search, FileText, Loader2 } from 'lucide-react';

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // In local dev, point this to localhost:7860 or your backend URL.
      // Once deployed to HF Spaces, replace with the deployed Space API URL.
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860';
      
      const response = await fetch(`${API_URL}/summarize/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, top_k: 2 }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch summary from server.');
      }

      const data = await response.json();
      setResults(data.summaries);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <div className="header">
        <h1 className="title">LexSumm</h1>
        <p className="subtitle">
          AI-powered legal document summarization. Enter a query or keywords to retrieve and summarize relevant legal texts instantly.
        </p>
      </div>

      <div className="search-box glass-panel">
        <div className="input-wrapper">
          <input 
            type="text" 
            className="search-input" 
            placeholder="e.g., lakshminarayana iyer dispute..." 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
        </div>
        <button 
          className="btn-primary" 
          onClick={handleSearch}
          disabled={loading || !query.trim()}
        >
          {loading ? (
             <>Searching <Loader2 className="animate-spin" size={20} /></>
          ) : (
             <>Search & Summarize <Search size={20} /></>
          )}
        </button>
      </div>

      {error && (
        <div className="glass-panel" style={{ padding: '1.5rem', color: '#f87171', border: '1px solid rgba(248, 113, 113, 0.3)' }}>
          <p><strong>Error:</strong> {error}</p>
        </div>
      )}

      {loading && (
        <div className="loader">
          <div className="loader-dot"></div>
          <div className="loader-dot"></div>
          <div className="loader-dot"></div>
        </div>
      )}

      {results && !loading && (
        <div className="results-container">
          <h3 style={{ fontSize: '1.4rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={24} color="#a78bfa" /> Results
          </h3>
          {results.map((summary, index) => (
            <div key={index} className="result-card glass-panel">
              <div className="result-header">
                <span className="result-badge">Summary {index + 1}</span>
              </div>
              <div className="result-content">
                {summary}
              </div>
            </div>
          ))}
          {results.length === 0 && (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: '#9ca3af' }}>
              No summaries could be generated for this query.
            </div>
          )}
        </div>
      )}
    </main>
  );
}
