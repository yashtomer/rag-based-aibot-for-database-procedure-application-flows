import React from 'react';
import { X } from 'lucide-react';

export default function CustomModal({ isOpen, onClose, title, children }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-md"
        onClick={onClose}
      />
      
      {/* Modal Dialog */}
      <div 
        className="relative w-full max-w-lg ds-glass-panel rounded-xl overflow-hidden z-10 border border-[var(--ds-border-strong)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--ds-border)]">
          <h2 className="ds-heading-sm">{title}</h2>
          <button 
            onClick={onClose}
            className="text-[var(--ds-text-faint)] hover:text-white transition-colors p-1 rounded-md hover:bg-white/5"
          >
            <X size={18} />
          </button>
        </div>
        
        {/* Body */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );
}
