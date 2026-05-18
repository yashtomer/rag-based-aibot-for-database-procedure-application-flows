import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Layout
import DashboardLayout from './layouts/DashboardLayout';

// Pages
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import IngestionPage from './pages/IngestionPage';
import ExplorerPage from './pages/ExplorerPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      const parsed = JSON.parse(savedUser);
      if (parsed.name === 'Alex Mercer') {
        parsed.name = 'Aeologic User';
        localStorage.setItem('user', JSON.stringify(parsed));
      }
      return parsed;
    }
    return null;
  });

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  // Synchronize theme with HTML document root attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleLogin = (userInfo) => {
    setUser(userInfo);
    localStorage.setItem('user', JSON.stringify(userInfo));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Login Route */}
        <Route 
          path="/login" 
          element={
            <LoginPage 
              user={user} 
              onLogin={handleLogin} 
              theme={theme} 
              onToggleTheme={toggleTheme} 
            />
          } 
        />

        {/* Protected Console Routes */}
        <Route 
          path="/" 
          element={
            <DashboardLayout 
              user={user} 
              onLogout={handleLogout} 
              theme={theme} 
              onToggleTheme={toggleTheme} 
            />
          }
        >
          {/* Index Page (Dashboard & Chat) */}
          <Route index element={<DashboardPage />} />
          
          {/* Ingestion & DB reflection */}
          <Route path="ingestion" element={<IngestionPage />} />
          
          {/* Schema Registry Explorer */}
          <Route path="explorer" element={<ExplorerPage />} />
          
          {/* Global Configurations */}
          <Route path="settings" element={<SettingsPage />} />

          {/* Catch-all Redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
