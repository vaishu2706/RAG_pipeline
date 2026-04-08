import { useState, useRef, useEffect } from 'react';
import './App.css';

const API = 'http://localhost:8000';

function Sidebar({ onStatus }) {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const uploadFile = async () => {
    if (!file) return;
    setLoading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
      const data = await res.json();
      onStatus(data.message || data.detail, !res.ok);
    } catch {
      onStatus('Upload failed', true);
    }
    setLoading(false);
  };

  const ingestUrl = async () => {
    if (!url) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/ingest-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      onStatus(data.message || data.detail, !res.ok);
      setUrl('');
    } catch {
      onStatus('URL ingest failed', true);
    }
    setLoading(false);
  };

  return (
    <div className="sidebar">
      <h2>🔮 RAG Pipeline</h2>

      <div>
        <h3>Upload File</h3>
        <div className="upload-area">
          <input type="file" accept=".pdf,.docx,.txt,.csv" onChange={e => setFile(e.target.files[0])} />
          <button className="btn btn-primary" onClick={uploadFile} disabled={!file || loading}>
            {loading ? 'Ingesting...' : 'Ingest File'}
          </button>
        </div>
      </div>

      <div>
        <h3>Ingest URL</h3>
        <div className="url-area">
          <input
            placeholder="https://example.com/page"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && ingestUrl()}
          />
          <button className="btn btn-primary" onClick={ingestUrl} disabled={!url || loading}>
            {loading ? 'Ingesting...' : 'Ingest URL'}
          </button>
        </div>
      </div>
    </div>
  );
}

function Chat() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! Upload documents and ask me anything about them.' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (!res.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: `⚠️ ${data.detail}` }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer, sources: data.sources, warnings: data.warnings }]);
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error reaching the server.' }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.content}
            {m.warnings?.length > 0 && (
              <div className="warnings">
                {m.warnings.map((w, j) => <span key={j}>⚠️ {w}</span>)}
              </div>
            )}
            {m.sources?.length > 0 && (
              <div className="sources">
                <strong>Sources:</strong>
                {m.sources.map((s, j) => (
                  <span key={j}>{s.source || 'document'}: {s.content.slice(0, 80)}…</span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="thinking">Thinking…</div>}
        <div ref={bottomRef} />
      </div>
      <div className="input-row">
        <input
          placeholder="Ask a question about your documents…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
        />
        <button className="btn btn-primary" onClick={send} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

function Evaluation() {
  const [samples, setSamples] = useState([{ question: '', ground_truth: '' }]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const updateSample = (i, field, val) => {
    setSamples(prev => prev.map((s, idx) => idx === i ? { ...s, [field]: val } : s));
  };

  const runEval = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(samples.filter(s => s.question && s.ground_truth)),
      });
      const data = await res.json();
      setResults(data.results);
    } catch {
      setError('Evaluation failed');
    }
    setLoading(false);
  };

  const METRICS = ['faithfulness', 'answer_relevancy', 'context_recall'];

  return (
    <div className="eval-container">
      <h3>RAGAS Evaluation</h3>
      <p style={{ fontSize: 13, color: '#94a3b8' }}>
        Metrics: Faithfulness · Answer Relevancy · Context Recall (Retrieval Quality)
      </p>

      {samples.map((s, i) => (
        <div key={i} className="eval-sample">
          <input
            placeholder="Question"
            value={s.question}
            onChange={e => updateSample(i, 'question', e.target.value)}
          />
          <input
            placeholder="Ground truth answer"
            value={s.ground_truth}
            onChange={e => updateSample(i, 'ground_truth', e.target.value)}
          />
        </div>
      ))}

      <div className="eval-actions">
        <button className="btn btn-secondary" onClick={() => setSamples(p => [...p, { question: '', ground_truth: '' }])}>
          + Add Sample
        </button>
        <button className="btn btn-primary" onClick={runEval} disabled={loading}>
          {loading ? 'Evaluating…' : 'Run Evaluation'}
        </button>
        {error && <span style={{ color: '#f87171', fontSize: 13 }}>{error}</span>}
      </div>

      {results && (
        <div className="eval-results">
          <h4>Results (avg across samples)</h4>
          {METRICS.map(m => {
            const avg = results.reduce((acc, r) => acc + (r[m] ?? 0), 0) / results.length;
            return (
              <div key={m} className="metric-row">
                <span>{m.replace(/_/g, ' ')}</span>
                <span className="metric-value">{isNaN(avg) ? 'N/A' : avg.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('chat');
  const [status, setStatus] = useState({ msg: '', error: false });

  const handleStatus = (msg, error = false) => {
    setStatus({ msg, error });
    setTimeout(() => setStatus({ msg: '', error: false }), 4000);
  };

  return (
    <div className="app">
      <Sidebar onStatus={handleStatus} />
      <div className="main">
        <div className="tabs">
          <div className={`tab ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}>💬 Chat</div>
          <div className={`tab ${tab === 'eval' ? 'active' : ''}`} onClick={() => setTab('eval')}>📊 Evaluate</div>
          {status.msg && (
            <span className={`status-msg ${status.error ? 'error' : ''}`} style={{ marginLeft: 'auto', padding: '14px 24px' }}>
              {status.msg}
            </span>
          )}
        </div>
        {tab === 'chat' ? <Chat /> : <Evaluation />}
      </div>
    </div>
  );
}
