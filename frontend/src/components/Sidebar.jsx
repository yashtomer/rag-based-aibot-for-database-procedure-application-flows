import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, 
  Database, 
  TableProperties, 
  Settings, 
  LogOut, 
  X, 
  Terminal
} from 'lucide-react';
import CustomModal from './CustomModal';

export default function Sidebar({ isOpen, onClose, user, onLogout, theme }) {
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const navigate = useNavigate();

  const logoSrc = theme === 'light' ? '/logo.svg' : '/logo-white.svg';

  const menuItems = [
    { name: 'Dashboard & Chat', path: '/', icon: LayoutDashboard, role: 'user' },
    { name: 'Database Ingestion', path: '/ingestion', icon: Database, role: 'admin' },
    { name: 'Schema Explorer', path: '/explorer', icon: TableProperties, role: 'user' },
    { name: 'System Settings', path: '/settings', icon: Settings, role: 'admin' },
  ];

  const handleLogout = () => {
    setShowLogoutConfirm(false);
    onLogout();
    navigate('/login');
  };

  const allowedItems = menuItems.filter(item => 
    item.role === 'user' || (user && user.role === 'admin')
  );

  return (
    <>
      {/* Mobile Sidebar Backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar Drawer Container */}
      <div className={`
        fixed top-0 bottom-0 left-0 z-50 w-72 bg-[var(--ds-bg)] border-r border-[var(--ds-border)]
        flex flex-col justify-between transition-transform duration-300 ease-in-out
        lg:translate-x-0 lg:static lg:z-auto
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Header / Logo */}
        <div>
          <div className="h-16 flex items-center justify-between px-6 border-b border-[var(--ds-border)]">
            <div className="flex items-center justify-start py-2">
              <img 
                src={logoSrc} 
                alt="Aeologic Logo" 
                className="h-11 w-auto object-contain filter drop-shadow-[0_0_15px_rgba(198,32,8,0.25)] transition-all duration-300" 
              />
            </div>
            
            {/* Mobile close button */}
            <button 
              onClick={onClose}
              className="lg:hidden text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] p-1 rounded hover:bg-[var(--ds-nav-active-bg)] cursor-pointer"
            >
              <X size={16} />
            </button>
          </div>

          {/* Navigation Section */}
          <div className="p-4">
            <span className="px-3 text-[9.5px] uppercase tracking-[0.2em] font-semibold text-[var(--ds-text-faint)]">
              Main Menu
            </span>
            <nav className="mt-4 flex flex-col gap-1.5 custom-scrollbar overflow-y-auto">
              {allowedItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    onClick={() => {
                      if (window.innerWidth < 1024) onClose();
                    }}
                    className={({ isActive }) => `
                      relative flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 group
                      ${isActive ? 'text-[var(--ds-text)] bg-[var(--ds-nav-active-bg)]' : 'text-[var(--ds-text-faint)] hover:text-[var(--ds-text)] hover:bg-[var(--ds-nav-active-bg)]'}
                    `}
                  >
                    {({ isActive }) => (
                      <>
                        <Icon 
                          size={17} 
                          className={`transition-colors group-hover:text-[var(--ds-brand)] ${isActive ? 'text-[var(--ds-brand)]' : 'text-inherit'}`} 
                        />
                        <span>{item.name}</span>
                        
                        {/* Red Active Indicator with Glow */}
                        {isActive && (
                          <motion.div
                            layoutId="sidebarActive"
                            className="absolute right-0 top-1/4 bottom-1/4 w-[3px] bg-[var(--ds-brand)] rounded-l-md"
                            style={{ boxShadow: '0 0 10px var(--ds-brand-glow)' }}
                            transition={{ type: 'spring', damping: 30, stiffness: 200 }}
                          />
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Bottom Profile & Sign Out */}
        {user && (
          <div className="p-4 border-t border-[var(--ds-border)] flex flex-col gap-3">
            {/* User Profile Mini-Card */}
            <div className="flex items-center gap-3 px-3 py-2 bg-[var(--ds-surface-2)] rounded-xl border border-[var(--ds-border)]">
              <div className="w-9 h-9 rounded-full bg-[var(--ds-brand)] flex items-center justify-center text-white font-bold text-sm shadow-md">
                {user.name.split(' ').map(n => n[0]).join('').toUpperCase()}
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-semibold truncate text-[var(--ds-text)]">{user.name}</span>
                <span className="text-[9px] uppercase tracking-wider text-[var(--ds-text-faint)] truncate">{user.role}</span>
              </div>
            </div>

            {/* Logout Trigger */}
            <button
              onClick={() => setShowLogoutConfirm(true)}
              className="w-full ds-btn-primary py-2.5 flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer"
            >
              <LogOut size={13} />
              Sign Out
            </button>
          </div>
        )}
      </div>

      {/* Logout Confirmation Modal */}
      <CustomModal
        isOpen={showLogoutConfirm}
        onClose={() => setShowLogoutConfirm(false)}
        title="Confirm Sign Out"
      >
        <div className="flex flex-col gap-5">
          <p className="ds-body text-center">
            Are you sure you want to log out of the Database Intelligence Bot console? Your session will be terminated.
          </p>
          <div className="flex justify-end gap-3">
            <button 
              onClick={() => setShowLogoutConfirm(false)}
              className="ds-btn-ghost py-2 text-xs cursor-pointer"
            >
              Cancel
            </button>
            <button 
              onClick={handleLogout}
              className="ds-btn-primary py-2 text-xs bg-[var(--ds-brand)] cursor-pointer"
            >
              Yes, Sign Out
            </button>
          </div>
        </div>
      </CustomModal>
    </>
  );
}
