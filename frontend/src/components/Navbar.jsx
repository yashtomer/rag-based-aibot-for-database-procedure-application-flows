import React from 'react';
import { Menu, Sun, Moon } from 'lucide-react';

export default function Navbar({ onToggleSidebar, theme, onToggleTheme, user }) {
  return (
    <nav className="sticky top-0 z-40 h-16 w-full flex items-center justify-between px-6 bg-[var(--ds-navbar-bg)] backdrop-blur-xl border-b border-[var(--ds-border)]">
      {/* Left: Mobile Hamburger */}
      <div className="flex items-center gap-4">
        <button 
          onClick={onToggleSidebar}
          className="lg:hidden text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] p-2 rounded-md hover:bg-[var(--ds-nav-active-bg)] cursor-pointer"
        >
          <Menu size={18} />
        </button>
        <span className="lg:hidden font-serif ds-heading-sm font-semibold tracking-wide">
          DB Bot
        </span>
      </div>

      {/* Right: Status, Theme, Profile */}
      <div className="flex items-center gap-6">
        {/* Status Pill */}
        <div className="flex items-center gap-2 px-3 py-1 bg-[var(--ds-brand-dim)] border border-[var(--ds-brand-glow)] rounded-full">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--ds-brand)] ds-pulse" />
          <span className="text-[10px] uppercase tracking-wider font-semibold text-[var(--ds-brand)]">
            System Live
          </span>
        </div>

        {/* Theme Toggle */}
        <button 
          onClick={onToggleTheme}
          className="text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] p-2 rounded-lg hover:bg-[var(--ds-nav-active-bg)] transition-all cursor-pointer"
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
        >
          {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
        </button>

        <div className="h-4 w-[1px] bg-[var(--ds-border)]" />

        {/* User Badge */}
        {user && (
          <div className="flex items-center gap-3">
            <div className="hidden md:flex flex-col text-right">
              <span className="text-xs font-semibold text-[var(--ds-text)]">{user.name}</span>
              <span className="text-[9px] uppercase tracking-wider text-[var(--ds-text-faint)]">{user.role}</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-[var(--ds-brand)] flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-[var(--ds-brand-glow)]">
              {user.name.split(' ').map(n => n[0]).join('').toUpperCase()}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
