import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

async function api(path, options = {}) {
  const response = await fetch(`/api/${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Request failed. Please try again.');
  return body;
}

const suggestions = ['What is the annual learning budget?', 'How do I report a lost laptop?', 'What are core collaboration hours?'];

function App() {
  const [documents, setDocuments] = useState([]);
  const [mode, setMode] = useState('connecting');
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [units, setUnits] = useState(0);
  const bottom = useRef(null);
  const file = useRef(null);

  async function refresh() {
    const [docs, health] = await Promise.all([api('documents'), api('health')]);
    setDocuments(docs); setMode(health.mode);
  }
  useEffect(() => { refresh().catch(e => { setError(e.message); setMode('unavailable'); }); }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy]);

  async function ask(value = question) {
    const text = value.trim();
    if (text.length < 3 || busy) return;
    setQuestion(''); setError(''); setBusy(true);
    setMessages(previous => [...previous, { role: 'user', text }]);
    try {
      const result = await api('query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text }) });
      setMessages(previous => [...previous, { role: 'assistant', ...result }]);
      setUnits(previous => previous + result.usage.units);
    } catch (e) { setError(e.message); setQuestion(text); }
    finally { setBusy(false); }
  }

  async function upload(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setError(''); setNotice(''); setUploading(true);
    try {
      const form = new FormData(); form.append('file', selected);
      const result = await api('ingest', { method: 'POST', body: form });
      setNotice(`${result.source}: ${result.unchanged ? 'already up to date' : `${result.chunks} chunks indexed`}.`);
      await refresh();
    } catch (e) { setError(e.message); }
    finally { setUploading(false); event.target.value = ''; }
  }

  return <div className="reading-room">
    <header className="room-header">
      <a className="brand" href="/" aria-label="Margin House home">Margin House<span>DOCUPILOT / READING ROOM</span></a>
      <details className="library-drawer">
        <summary className="library-toggle"><span aria-hidden="true">☷</span> Library <span className="count">{documents.length}</span><span className="drawer-chevron" aria-hidden="true">+</span></summary>
        <aside className="library-panel" aria-label="Document library">
      <div className="eyebrow">THE COLLECTION</div>
      <h2>Your source material.</h2>
      <p className="library-description">A place for the documents behind your answers.</p>
      <div className="library-heading"><span>Knowledge library</span><span className="count">{documents.length}</span></div>
      <div className="document-list">{documents.map(doc => <div className="document" key={doc.source}><span className="file-icon">≡</span><div title={doc.source}>{doc.source}<small>{doc.chunks} sections indexed</small></div><span className="indexed-mark" title="Indexed">✓</span></div>)}</div>
      {!documents.length && <p className="empty-library">Upload a document or run the demo ingest to build your library.</p>}
      <input ref={file} className="visually-hidden" type="file" accept=".md,.txt,text/plain,text/markdown" onChange={upload} disabled={uploading} aria-label="Upload document" />
      <button className="upload-button" onClick={() => file.current.click()} disabled={uploading}>{uploading ? 'Indexing…' : '+ Add a document'}</button>
      <p className="upload-hint">Markdown or TXT · up to 1 MB<br/>Same filename replaces the document.</p>
      {notice && <p className="upload-notice" role="status">{notice}</p>}
      <div className="library-footer">Margin House<small>Read closely. Follow the evidence.</small></div>
        </aside>
      </details>
    </header>
    <main>
      <div className="room-caption"><span>A LITTLE SPACE TO THINK</span><span className={`mode ${mode === 'mock' ? 'online' : ''}`}>{mode === 'mock' ? 'Offline mock' : mode === 'provider' ? 'Live provider' : mode}</span></div>
      <div className={`conversation ${messages.length ? 'has-messages' : ''}`}>
        <div className="intro"><div className="eyebrow">READ. ASK. UNDERSTAND.</div><h1>Good questions.<br/><em>Well-read answers.</em></h1><p>Bring a question to your library.<br/>Find an answer with the evidence in the margins.</p></div>
        {!messages.length && <><div className="suggestion-label">A FEW PLACES TO START</div><div className="suggestions">{suggestions.map((text, index) => <button key={text} onClick={() => ask(text)} disabled={busy}><span className="suggestion-number">0{index + 1}</span>{text}<span className="arrow">↗</span></button>)}</div></>}
        <div className="messages" aria-live="polite" aria-busy={busy}>{messages.map((message, index) => <article className={`message ${message.role}`} key={index}><div className="message-label">{message.role === 'user' ? 'YOUR QUESTION' : 'DOCUPILOT / READING NOTES'}{message.refused && <span className="refusal-tag">Insufficient evidence</span>}</div><div className="answer">{message.text || message.answer}</div>{message.citations?.length > 0 && <div className="citations"><div className="sources-label">{message.citations.length} SOURCES · OPEN TO VERIFY</div>{message.citations.map((citation, i) => <details key={`${citation.chunk_id}-${i}`}><summary><span className="citation-number">{i + 1}</span><span className="citation-content"><strong>{citation.source}</strong><small>{citation.section}</small><span className="citation-preview">“{citation.quote}”</span><span className="source-action">Read source excerpt <span aria-hidden="true">↗</span></span></span><span className="expand" aria-hidden="true">+</span></summary><blockquote>{citation.quote}</blockquote><div className="citation-meta">Similarity {Math.round(citation.score * 100)}% · Chunk {citation.chunk_id}</div></details>)}</div>}{message.usage && <div className="request-cost">{message.usage.units} {message.usage.unit.replace('_', ' ')}</div>}</article>)}{busy && <div className="loading" role="status"><span/> Finding the evidence…</div>}<div ref={bottom}/></div>
      </div>
      <div className="composer-area">{error && <div className="error" role="alert">{error} <button onClick={() => { setError(''); refresh().catch(e => setError(e.message)); }}>Reconnect</button></div>}<div className="source-mode"><span className="source-mode-icon" aria-hidden="true">▤</span> Answering from your library <span>{documents.length} documents</span></div><form onSubmit={event => { event.preventDefault(); ask(); }}><label className="visually-hidden" htmlFor="question">Ask a question</label><textarea id="question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask something worth reading about…" rows={1} maxLength={2000} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); ask(); } }} /><button className="send" type="submit" disabled={busy || question.trim().length < 3} aria-label="Send question">↑</button></form><div className="composer-footer"><span>Grounded in your documents. Built to say “I don’t know”.</span><span>{units} {mode === 'provider' ? 'tokens' : 'mock units'} this session</span></div></div>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
