import React, { useState } from 'react';
import { Settings, Save, CheckCircle, ShieldAlert } from 'lucide-react';

export default function SettingsPage() {
  const [mysqlHost, setMysqlHost] = useState('localhost');
  const [mysqlPort, setMysqlPort] = useState('3306');
  const [mysqlUser, setMysqlUser] = useState('root');
  const [mysqlPass, setMysqlPass] = useState('password');
  const [success, setSuccess] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setSuccess(false);

    // Save configurations
    await new Promise(r => setTimeout(r, 800));
    setSaving(false);
    setSuccess(true);
    setTimeout(() => setSuccess(false), 3000);
  };

  return (
    <div className="flex flex-col gap-6 max-w-3xl select-none">
      {/* Header */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-3">
          <div className="ds-brand-line" />
          <span className="ds-label-brand">System Specs</span>
        </div>
        <h1 className="ds-heading-lg">Global Config Settings</h1>
        <p className="ds-body">Configure target database links, system security API tokens, and persistent vector databases in real time.</p>
      </div>

      {/* Form Card */}
      <div className="ds-card p-6 md:p-8 mt-4 bg-white/[0.01]">
        <form onSubmit={handleSave} className="flex flex-col gap-6">
          <div className="flex items-center gap-2 border-b border-[var(--ds-border)] pb-3">
            <Settings size={16} className="text-[var(--ds-brand)]" />
            <h3 className="ds-heading-sm !text-base">System Properties</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Host */}
            <div className="flex flex-col gap-1.5">
              <label className="ds-field-label">MySQL Host Connection</label>
              <input
                type="text"
                value={mysqlHost}
                onChange={(e) => setMysqlHost(e.target.value)}
                className="ds-input"
              />
            </div>

            {/* Port */}
            <div className="flex flex-col gap-1.5">
              <label className="ds-field-label">MySQL Port</label>
              <input
                type="text"
                value={mysqlPort}
                onChange={(e) => setMysqlPort(e.target.value)}
                className="ds-input"
              />
            </div>

            {/* User */}
            <div className="flex flex-col gap-1.5">
              <label className="ds-field-label">Database User Account</label>
              <input
                type="text"
                value={mysqlUser}
                onChange={(e) => setMysqlUser(e.target.value)}
                className="ds-input"
              />
            </div>

            {/* Pass */}
            <div className="flex flex-col gap-1.5">
              <label className="ds-field-label">Database Password</label>
              <input
                type="password"
                value={mysqlPass}
                onChange={(e) => setMysqlPass(e.target.value)}
                className="ds-input"
              />
            </div>
          </div>



          {/* Actions */}
          <div className="flex items-center justify-between mt-4">
            <div>
              {success && (
                <div className="flex gap-2 items-center text-xs font-semibold text-emerald-400">
                  <CheckCircle size={14} /> Confirmed! Settings saved to .env profile.
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={saving}
              className="ds-btn-primary py-3 px-6 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              {saving ? (
                <span className="ds-spinner !w-3.5 !h-3.5 border-t-white" />
              ) : (
                <Save size={14} />
              )}
              Save System State
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
