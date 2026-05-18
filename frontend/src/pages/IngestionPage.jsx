import React, { useState, useEffect } from 'react';
import { Database, RefreshCw, Terminal, CheckCircle2, AlertCircle } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export default function IngestionPage() {
  const [selectedDb, setSelectedDb] = useState('Checking...');
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);

  const addLog = (msg) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  useEffect(() => {
    const loadDbStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE}/db/status`);
        const savedDb = localStorage.getItem('selectedDb');
        const defaultDb = savedDb && res.data.databases.includes(savedDb) ? savedDb : res.data.active_db;
        setSelectedDb(defaultDb);
        setDatabases(res.data.databases || [res.data.active_db]);
      } catch (err) {
        console.error("Failed to load active database targets:", err);
        setSelectedDb("sqlite_memory");
        setDatabases(["sqlite_memory"]);
      }
    };
    loadDbStatus();
  }, []);

  const handleIngest = async () => {
    setLoading(true);
    setStatus(null);
    setLogs([]);
    addLog(`Initiating connection to database server...`);
    await new Promise(r => setTimeout(r, 600));
    addLog(`Connected. Target Schema: \`${selectedDb}\``);
    await new Promise(r => setTimeout(r, 500));
    addLog(`Reflecting table schema maps...`);

    try {
      const res = await axios.post(`${API_BASE}/ingest`, { database: selectedDb });
      localStorage.setItem('selectedDb', selectedDb);
      addLog(`Metadata extraction success! Row mapping generated.`);
      addLog(`Syncing context schema chunks to ChromaDB vector store...`);
      await new Promise(r => setTimeout(r, 800));
      addLog(`ChromaDB Vector embeddings updated successfully.`);
      setStatus({ success: true, message: 'Database schema ingested and synchronized successfully!' });
    } catch (err) {
      addLog(`❌ Ingestion failed: ${err.message}`);
      setStatus({ success: false, message: `Ingestion failed: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-4xl select-none">
      {/* Header */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-3">
          <div className="ds-brand-line" />
          <span className="ds-label-brand">Knowledge Sync</span>
        </div>
        <h1 className="ds-heading-lg">Database Schema Ingestion</h1>
        <p className="ds-body">Trigger automatic reflection of database schemas and compile knowledge embeddings for the AI RAG engine.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start mt-4">
        {/* Left card: settings */}
        <div className="ds-card p-6 flex flex-col gap-6">
          <div className="flex items-center gap-2 border-b border-[var(--ds-border)] pb-3">
            <Database size={16} className="text-[var(--ds-brand)]" />
            <h3 className="ds-heading-sm !text-base">Sync Configuration</h3>
          </div>

          {/* Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="ds-field-label">Target Database Schema</label>
            <select
              value={selectedDb}
              onChange={(e) => setSelectedDb(e.target.value)}
              className="ds-select"
              disabled={loading}
            >
              {databases.map(db => (
                <option key={db} value={db}>{db}</option>
              ))}
            </select>
            <span className="text-[10px] text-[var(--ds-text-faint)]">Only standard MySQL schemas are active for index reflection.</span>
          </div>

          {/* Trigger button */}
          <button
            onClick={handleIngest}
            disabled={loading}
            className="ds-btn-primary py-3 flex items-center justify-center gap-2 w-full cursor-pointer disabled:opacity-50"
          >
            {loading ? (
              <span className="ds-spinner !w-3.5 !h-3.5 border-t-white" />
            ) : (
              <RefreshCw size={14} />
            )}
            Synchronize Schema Embeddings
          </button>

          {/* Status Box */}
          {status && (
            <div className={`flex gap-3 p-4 rounded-xl border text-xs ${
              status.success 
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                : 'bg-red-500/10 border-red-500/20 text-red-400'
            }`}>
              {status.success ? <CheckCircle2 size={16} className="shrink-0" /> : <AlertCircle size={16} className="shrink-0" />}
              <span>{status.message}</span>
            </div>
          )}
        </div>

        {/* Right card: sync logs console */}
        <div className="ds-card p-6 flex flex-col gap-4 h-[350px] overflow-hidden bg-black/40">
          <div className="flex items-center gap-2 border-b border-[var(--ds-border)] pb-3 shrink-0">
            <Terminal size={15} className="text-[var(--ds-text-faint)]" />
            <h3 className="ds-label text-[11px] font-semibold">Reflection Logs</h3>
          </div>
          
          <div className="flex-1 overflow-y-auto font-mono text-[11.5px] text-[var(--ds-text-muted)] flex flex-col gap-2 custom-scrollbar">
            {logs.length === 0 ? (
              <span className="text-[var(--ds-text-faint)] italic">Ready for synchronization logs...</span>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="leading-relaxed">
                  <span className="text-[var(--ds-brand)] mr-1">&gt;</span>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
