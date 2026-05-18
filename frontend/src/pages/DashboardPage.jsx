import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  Send, 
  Database, 
  Settings, 
  Cpu, 
  CheckCircle, 
  AlertCircle, 
  Activity, 
  RefreshCw,
  Terminal,
  Grid,
  ChevronRight
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

// Isolated Mermaid Diagram Renderer using iframe and CDN
function MermaidRenderer({ code }) {
  const [showCode, setShowCode] = useState(false);
  const uid = 'm' + Math.random().toString(36).substring(2, 10);
  const escapedCode = code
    .replace(/\\/g, '\\\\')
    .replace(/`/g, '\\`')
    .replace(/\$/g, '\\$');

  const srcDoc = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { margin: 0; padding: 10px; display: flex; justify-content: center; background: transparent; overflow: auto; }
        #container { width: 100%; display: flex; justify-content: center; }
      </style>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: false, theme: 'dark' });
        const el = document.getElementById('render-el');
        try {
          const { svg } = await mermaid.render('${uid}_svg', el.textContent);
          el.innerHTML = svg;
        } catch (e) {
          el.innerHTML = '<pre style="color:#F87171; font-family: monospace; font-size:12px;">Diagram rendering error: ' + e.message + '</pre>';
        }
      </script>
    </head>
    <body>
      <div id="container">
        <div id="render-el">${escapedCode}</div>
      </div>
    </body>
    </html>
  `;

  return (
    <div className="w-full mt-4 flex flex-col gap-2">
      <div className="relative w-full border border-[var(--ds-border)] rounded-xl overflow-hidden bg-black/40">
        {/* Header toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--ds-border)] bg-white/5">
          <span className="text-[10px] uppercase tracking-wider font-semibold text-[var(--ds-brand)] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--ds-brand)]" />
            Generated ER Diagram
          </span>
          <button
            onClick={() => setShowCode(!showCode)}
            className="text-[10.5px] text-[var(--ds-text-faint)] hover:text-white px-2 py-0.5 border border-[var(--ds-border)] rounded bg-white/5 cursor-pointer"
          >
            {showCode ? 'Hide Source' : 'View Source'}
          </button>
        </div>

        {/* Diagram Display */}
        <div className="p-4 flex justify-center overflow-auto min-h-[300px]">
          <iframe
            srcDoc={srcDoc}
            title="Mermaid Diagram"
            className="w-full border-none min-h-[350px]"
            sandbox="allow-scripts"
          />
        </div>
      </div>

      {showCode && (
        <pre className="p-4 rounded-xl border border-[var(--ds-border)] bg-black/50 text-xs font-mono text-[var(--ds-text-muted)] overflow-x-auto">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Hello! I am your Database Intelligence Bot. Ask me anything about your database schema, table structures, or request an entity relationship diagram (ERD).' 
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dbStatus, setDbStatus] = useState('Checking...');
  const [activeDb, setActiveDb] = useState('Checking...');
  const [tableCount, setTableCount] = useState(0);
  
  // Model Configurations
  const modelsMap = {
    'Gemini (Google)': ['gemini-3.1-pro-preview', 'gemini-3.1-flash-lite', 'gemini-2.5-pro', 'gemini-2.5-flash', 'Custom Model...'],
    'Groq (Fast Inference)': ['llama-4-70b', 'llama-4-8b', 'deepseek-r1-distill-llama-70b', 'Custom Model...'],
    'OpenAI': ['gpt-5.4-pro', 'gpt-5.4-thinking', 'gpt-5.4-mini', 'Custom Model...'],
    'Anthropic': ['claude-sonnet-4-6', 'claude-opus-4-6', 'claude-sonnet-3-7', 'Custom Model...']
  };

  const [provider, setProvider] = useState('Gemini (Google)');
  const [model, setModel] = useState('gemini-3.1-pro-preview');
  const [customModelName, setCustomModelName] = useState('');
  const [isCustomModel, setIsCustomModel] = useState(false);
  
  // API Key State (loaded strictly from sessionStorage)
  const [apiKey, setApiKey] = useState(() => {
    return sessionStorage.getItem(`api_key_Gemini (Google)`) || '';
  });

  const handleProviderChange = (newProvider) => {
    setProvider(newProvider);
    const defaultModel = modelsMap[newProvider][0];
    setModel(defaultModel);
    setIsCustomModel(false);
    
    const savedKey = sessionStorage.getItem(`api_key_${newProvider}`) || '';
    setApiKey(savedKey);
  };

  const handleModelChange = (val) => {
    if (val === 'Custom Model...') {
      setIsCustomModel(true);
      setModel('Custom Model...');
    } else {
      setIsCustomModel(false);
      setModel(val);
    }
  };

  const handleApiKeyChange = (e) => {
    const val = e.target.value;
    setApiKey(val);
    sessionStorage.setItem(`api_key_${provider}`, val);
  };

  const [testStatus, setTestStatus] = useState(null);
  const [testingModel, setTestingModel] = useState(false);

  const chatBottomRef = useRef(null);

  const exampleQueries = [
    "What tables exist and how do they relate?",
    "Describe the customer order tables and foreign keys",
    "Show ER diagram for the inventory module",
    "What is the column layout of the orders table?"
  ];

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Initial check database backend status
  useEffect(() => {
    const checkDb = async () => {
      try {
        const savedDb = localStorage.getItem('selectedDb') || '';
        const url = savedDb ? `${API_BASE}/db/status?database=${savedDb}` : `${API_BASE}/db/status`;
        const res = await axios.get(url);
        setActiveDb(res.data.active_db);
        setDbStatus(res.data.status);
        setTableCount(res.data.tables.length);
        if (res.data.active_db && res.data.active_db !== 'sqlite_memory') {
          localStorage.setItem('selectedDb', res.data.active_db);
        }
      } catch (err) {
        console.error("Failed to fetch database connection status:", err);
        setActiveDb("None");
        setDbStatus("Offline");
        setTableCount(0);
      }
    };
    checkDb();
  }, []);

  const handleSend = async (textToSend) => {
    const prompt = textToSend || input;
    if (!prompt.trim()) return;

    if (!textToSend) setInput('');

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: prompt }]);
    setLoading(true);

    const isDiagram = /diagram|erd|visualize|schema diagram|draw|relationship/i.test(prompt);
    const resolvedModel = isCustomModel ? customModelName : model;

    try {
      if (isDiagram) {
        // Send request to /diagram
        const res = await axios.post(`${API_BASE}/diagram`, { 
          request: prompt,
          provider: provider,
          model: resolvedModel,
          api_key: apiKey || null
        });
        const mermaidCode = res.data.mermaid_code;
        
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: "Here is the ER diagram mapped from your database context:",
          isDiagram: true,
          diagramCode: mermaidCode
        }]);
      } else {
        // Send request to /chat
        const res = await axios.post(`${API_BASE}/chat`, { 
          query: prompt,
          provider: provider,
          model: resolvedModel,
          api_key: apiKey || null
        });
        setMessages(prev => [...prev, { role: 'assistant', content: res.data.answer }]);
      }
    } catch (err) {
      console.error(err);
      const detailMsg = err.response?.data?.detail;
      const displayMsg = detailMsg 
        ? `⚠️ Configuration Alert:\n\n${detailMsg}\n\nPlease enter your API Session Key in the "Core AI Model Configuration" card on the right.`
        : `❌ Error communicating with LLM Node. Please ensure your backend container is running at ${API_BASE}. \n\nDetail: ${err.message}`;
        
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: displayMsg 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnectivity = async () => {
    setTestingModel(true);
    setTestStatus(null);
    const resolvedModel = isCustomModel ? customModelName : model;
    try {
      const res = await axios.post(`${API_BASE}/test-connection`, {
        provider: provider,
        model: resolvedModel,
        api_key: apiKey || null
      });
      if (res.data.success) {
        setTestStatus({ success: true, message: res.data.message });
      } else {
        setTestStatus({ success: false, message: res.data.message });
      }
    } catch (e) {
      console.error(e);
      setTestStatus({ 
        success: false, 
        message: `Validation Error: ${e.response?.data?.detail || e.message}` 
      });
    } finally {
      setTestingModel(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full min-h-0">
      
      {/* LEFT COLUMN: System Monitor & Control Room */}
      <div className="lg:col-span-4 flex flex-col gap-6 h-full overflow-y-auto pr-1 select-none">
        
        {/* Section Header */}
        <div className="flex items-center gap-3">
          <div className="ds-brand-line" />
          <span className="ds-label-brand">Control Panel</span>
        </div>

        {/* Technical Stats Dashboard */}
        <div className="grid grid-cols-2 gap-4">
          {/* Stat Item 1 */}
          <div className="ds-card p-4 flex flex-col gap-1 border-l-2 border-l-[var(--ds-brand)]">
            <span className="ds-label text-[9.5px]">Active DB</span>
            <span className="font-serif text-xl font-bold text-[var(--ds-text)]">{activeDb}</span>
            <span className="text-[10px] text-[var(--ds-success)] flex items-center gap-1 mt-1 font-semibold">
              <CheckCircle size={10} /> Active Node
            </span>
          </div>

          {/* Stat Item 2 */}
          <div className="ds-card p-4 flex flex-col gap-1">
            <span className="ds-label text-[9.5px]">Ingested Tables</span>
            <span className="font-serif text-xl font-bold text-[var(--ds-text)]">{tableCount}</span>
            <span className="text-[10px] text-[var(--ds-text-faint)] mt-1 font-medium">
              ChromaDB Sync
            </span>
          </div>

          {/* Stat Item 3 */}
          <div className="ds-card p-4 flex flex-col gap-1">
            <span className="ds-label text-[9.5px]">API Latency</span>
            <span className="font-serif text-xl font-bold text-[var(--ds-text)]">124ms</span>
            <span className="text-[10px] text-[var(--ds-text-faint)] mt-1 flex items-center gap-1 font-medium">
              <Activity size={10} className="text-[var(--ds-brand)]" /> Standard Node
            </span>
          </div>

          {/* Stat Item 4 */}
          <div className="ds-card p-4 flex flex-col gap-1 border-l-2 border-l-[var(--ds-success)]">
            <span className="ds-label text-[9.5px]">DB Status</span>
            <span className="font-serif text-xl font-bold text-[var(--ds-text)]">{dbStatus}</span>
            <span className="text-[10px] text-[var(--ds-success)] flex items-center gap-1 mt-1 font-semibold">
              <CheckCircle size={10} /> Live Connection
            </span>
          </div>
        </div>

        {/* Model Selector Card */}
        <div className="ds-card p-6 flex flex-col gap-5 bg-white/[0.01]">
          <div className="flex items-center gap-2 border-b border-[var(--ds-border)] pb-3">
            <Cpu size={16} className="text-[var(--ds-brand)]" />
            <h3 className="ds-heading-sm !text-[16px]">Core AI Model Configuration</h3>
          </div>

          {/* Provider Select */}
          <div className="flex flex-col gap-1.5">
            <label className="ds-field-label">LLM Provider</label>
            <select
              value={provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              className="ds-select"
            >
              <option value="Gemini (Google)">Gemini (Google)</option>
              <option value="Groq (Fast Inference)">Groq (Fast Inference)</option>
              <option value="OpenAI">OpenAI (GPT-4)</option>
              <option value="Anthropic">Anthropic (Claude)</option>
            </select>
          </div>

          {/* Model Select */}
          <div className="flex flex-col gap-1.5">
            <label className="ds-field-label">Active Model</label>
            <select
              value={model}
              onChange={(e) => handleModelChange(e.target.value)}
              className="ds-select"
            >
              {(modelsMap[provider] || []).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* Custom Model Name Input */}
          {isCustomModel && (
            <div className="flex flex-col gap-1.5 animate-fadeIn">
              <label className="ds-field-label">Custom Model Name</label>
              <input
                type="text"
                placeholder="e.g. claude-3-opus-20240229"
                value={customModelName}
                onChange={(e) => setCustomModelName(e.target.value)}
                className="ds-input py-2 text-xs font-semibold focus:border-[var(--ds-brand)] bg-[var(--ds-surface-2)]"
              />
            </div>
          )}

          {/* Secure Session API Key Input */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <label className="ds-field-label">API Session Key</label>
              <span className="text-[9px] uppercase tracking-wider font-bold text-[var(--ds-text-faint)] bg-white/5 border border-white/10 rounded px-1.5 py-0.5">Session Only</span>
            </div>
            <input
              type="password"
              placeholder={apiKey ? "••••••••••••••••••••" : `Enter ${provider} API key (overrides .env)...`}
              value={apiKey}
              onChange={handleApiKeyChange}
              className="ds-input py-2 text-xs font-semibold focus:border-[var(--ds-brand)] bg-[var(--ds-surface-2)]"
            />
          </div>

          {/* Test connection Button */}
          <button
            onClick={handleTestConnectivity}
            disabled={testingModel}
            className="ds-btn-ghost w-full py-2.5 flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer"
          >
            {testingModel ? (
              <span className="ds-spinner !w-3.5 !h-3.5 border-t-[var(--ds-brand)]" />
            ) : (
              <RefreshCw size={12} />
            )}
            Verify API Link
          </button>

          {/* Test Status feedback */}
          {testStatus && (
            <div className={`flex gap-2 p-3 text-[11px] border rounded-lg ${
              testStatus.success 
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                : 'bg-red-500/10 border-red-500/20 text-red-400'
            }`}>
              {testStatus.success ? <CheckCircle size={14} className="shrink-0" /> : <AlertCircle size={14} className="shrink-0" />}
              <span>{testStatus.message}</span>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: RAG Chat Node */}
      <div className="lg:col-span-8 flex flex-col h-full overflow-hidden border border-[var(--ds-border)] bg-black/15 rounded-2xl relative">
        
        {/* Chat Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--ds-border)] bg-[var(--ds-surface)]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[var(--ds-brand-dim)] border border-[var(--ds-brand-glow)] flex items-center justify-center text-[var(--ds-brand)]">
              <Bot size={18} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-[var(--ds-text)]">RAG Chat Engine</h2>
              <p className="text-[10px] text-[var(--ds-text-faint)] flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-[var(--ds-success)] rounded-full" />
                Linked to {activeDb} context schema
              </p>
            </div>
          </div>
        </div>

        {/* Message Thread */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6 custom-scrollbar">
          {messages.map((msg, index) => (
            <div 
              key={index}
              className={`flex gap-4 max-w-[85%] ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}
            >
              {/* Avatar */}
              <div className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center font-bold text-xs ${
                msg.role === 'user' 
                  ? 'bg-white/10 text-white' 
                  : 'bg-[var(--ds-brand-dim)] border border-[var(--ds-brand-glow)] text-[var(--ds-brand)]'
              }`}>
                {msg.role === 'user' ? 'U' : <Bot size={15} />}
              </div>

              {/* Card content */}
              <div className={`ds-card p-4 rounded-xl relative ${
                msg.role === 'user' 
                  ? '!bg-[var(--ds-surface-3)] border-[var(--ds-border-strong)] text-[var(--ds-text)]' 
                  : 'text-[var(--ds-text-muted)]'
              }`}>
                <p className="ds-body !text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                
                {/* Embedded Diagram */}
                {msg.isDiagram && msg.diagramCode && (
                  <MermaidRenderer code={msg.diagramCode} />
                )}
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {loading && (
            <div className="flex gap-4 max-w-[85%]">
              <div className="w-8 h-8 rounded-lg shrink-0 flex items-center justify-center bg-[var(--ds-brand-dim)] border border-[var(--ds-brand-glow)] text-[var(--ds-brand)]">
                <Bot size={15} />
              </div>
              <div className="ds-card p-4 rounded-xl flex items-center gap-3">
                <span className="ds-spinner-brand" />
                <span className="text-xs text-[var(--ds-text-faint)]">Bot is searching database schema...</span>
              </div>
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* Example Queries section */}
        {messages.length === 1 && !loading && (
          <div className="px-6 pb-2">
            <span className="ds-label text-[9px] mb-2 block">💡 Example Schema Queries:</span>
            <div className="grid grid-cols-2 gap-3">
              {exampleQueries.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(q)}
                  className="px-4 py-2.5 rounded-lg border border-[var(--ds-border)] bg-white/[0.01] hover:bg-white/[0.03] text-xs font-semibold text-left text-[var(--ds-text-muted)] hover:text-[var(--ds-text)] transition-colors flex justify-between items-center group cursor-pointer"
                >
                  <span className="truncate mr-2">{q}</span>
                  <ChevronRight size={14} className="shrink-0 text-[var(--ds-text-faint)] group-hover:text-[var(--ds-brand)] transition-colors" />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Form */}
        <div className="p-4 border-t border-[var(--ds-border)] bg-[var(--ds-surface)]">
          <form 
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex gap-3"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about table layouts, primary keys, relationships, or draw an ER diagram..."
              className="ds-input flex-1 py-3 px-4 focus:border-[var(--ds-brand)] rounded-lg text-sm bg-[var(--ds-surface-2)]"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="ds-btn-primary px-5 py-3 rounded-lg flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={14} />
              <span className="hidden md:inline">Execute</span>
            </button>
          </form>
        </div>

      </div>

    </div>
  );
}
