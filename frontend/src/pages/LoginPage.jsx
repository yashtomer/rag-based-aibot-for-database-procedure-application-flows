import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mail, Lock, Terminal, ShieldAlert, Sun, Moon, Eye, EyeOff } from 'lucide-react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export default function LoginPage({ user, onLogin, theme, onToggleTheme }) {
  const logoSrc = theme === 'light' ? '/logo.svg' : '/logo-white.svg';
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      navigate('/');
    }
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const res = await axios.post(`${API_BASE}/auth/login`, { email, password });
      onLogin(res.data);
      navigate('/');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Invalid credentials or unable to connect to auth server.');
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[var(--ds-bg)] overflow-hidden relative">
      {/* Floating Theme Toggle in Top Right */}
      <div className="absolute top-6 right-6 z-20">
        <button 
          onClick={onToggleTheme}
          className="text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] p-2.5 rounded-lg border border-[var(--ds-border)] bg-[var(--ds-surface-2)] hover:bg-[var(--ds-nav-active-bg)] transition-all cursor-pointer shadow-md flex items-center justify-center"
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
        >
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
      </div>
      {/* Left Panel: Architectural Blueprint / Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative h-full ds-grid-bg items-center justify-center border-r border-[var(--ds-border)]">
        {/* Glow Sphere */}
        <div className="absolute inset-0 bg-gradient-to-tr from-[var(--ds-ambient-color)] to-transparent pointer-events-none" />
        
        {/* Corner Decor Brackets */}
        <div className="absolute top-12 left-12 w-6 h-6 border-t-2 border-l-2 border-[var(--ds-border-strong)]" />
        <div className="absolute top-12 right-12 w-6 h-6 border-t-2 border-r-2 border-[var(--ds-border-strong)]" />
        <div className="absolute bottom-12 left-12 w-6 h-6 border-b-2 border-l-2 border-[var(--ds-border-strong)]" />
        <div className="absolute bottom-12 right-12 w-6 h-6 border-b-2 border-r-2 border-[var(--ds-border-strong)]" />

        {/* Blueprint Brand Title */}
        <div className="relative z-10 max-w-lg px-8">
          <div className="flex items-center gap-3 mb-6">
            <img src={logoSrc} alt="Aeologic Logo" className="h-14 w-auto filter drop-shadow-[0_0_15px_rgba(198,32,8,0.3)]" />
            <div className="h-[2px] w-24 bg-[var(--ds-brand)]" />
            <span className="text-xs uppercase tracking-[0.2em] font-semibold text-[var(--ds-brand)]">System Console</span>
          </div>
          
          <h1 className="ds-display mb-4">
            Database<br />
            <span className="text-[var(--ds-brand)]">Intelligence</span> Bot
          </h1>
          
          <p className="ds-body text-base max-w-sm text-[var(--ds-text-muted)]">
            A secure RAG-based engine powered by LLM and ChromaDB to analyze schemas, reflect structure, and generate high-fidelity technical documentation.
          </p>
        </div>
      </div>

      {/* Right Panel: Sleek Login Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-[var(--ds-bg)] relative">
        {/* Radial ambient glow behind card */}
        <div className="absolute inset-0 bg-radial-gradient(ellipse at 50% 50%, var(--ds-ambient-color) 0%, transparent 60%) pointer-events-none opacity-50" />

        {/* Form Container */}
        <div className="w-full max-w-md relative z-10">
          <div className="mb-8 text-center lg:text-left">
            <span className="ds-label-brand">Security Portal</span>
            <h2 className="ds-heading-lg mt-2">Initialize Session</h2>
            <p className="ds-caption mt-1">Authenticate to access the intelligence node</p>
          </div>

          {/* Login Card */}
          <div className="ds-card-lg p-8 md:p-10 ds-glass-panel">
            <form onSubmit={handleSubmit} className="flex flex-col gap-6">
              {/* Error Box */}
              {error && (
                <div className="flex gap-3 p-4 bg-[var(--ds-brand-dim)] border border-[var(--ds-brand-glow)] text-xs text-[#F87171] rounded-lg">
                  <ShieldAlert size={16} className="shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {/* Email Input (Line-style) */}
              <div className="flex flex-col gap-1.5">
                <label className="ds-field-label">Email Address</label>
                <div className="ds-input-wrap">
                  <Mail size={16} className="ds-input-icon" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@aeologic.com"
                    className="ds-input-line"
                  />
                </div>
              </div>

              {/* Password Input (Line-style) */}
              <div className="flex flex-col gap-1.5">
                <label className="ds-field-label">Password</label>
                <div className="ds-input-wrap">
                  <Lock size={16} className="ds-input-icon" />
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="ds-input-line pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] transition-colors cursor-pointer"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>



              {/* Submit Button */}
              <button
                type="submit"
                className="w-full ds-btn-primary py-3.5 mt-2 cursor-pointer"
              >
                Access Node
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
