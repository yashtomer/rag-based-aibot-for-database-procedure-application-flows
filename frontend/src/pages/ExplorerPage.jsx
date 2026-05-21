import React, { useState, useEffect } from 'react';
import { Table2, Search, Eye, Key, Loader2, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export default function ExplorerPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [databases, setDatabases] = useState([]);
  const [selectedDb, setSelectedDb] = useState(() => localStorage.getItem('selectedDb') || '');
  const [tablesList, setTablesList] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [currentTable, setCurrentTable] = useState(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Discover actual tables inside the selected database schema
  useEffect(() => {
    const fetchTables = async () => {
      try {
        setLoadingList(true);
        const savedDb = localStorage.getItem('selectedDb');
        const currentSelected = selectedDb || savedDb || '';
        const url = currentSelected 
          ? `${API_BASE}/db/status?database=${currentSelected}` 
          : `${API_BASE}/db/status`;
        const res = await axios.get(url);
        
        const tables = res.data.tables || [];
        setTablesList(tables);
        setDatabases(res.data.databases || []);
        
        if (!selectedDb) {
          setSelectedDb(res.data.active_db);
          localStorage.setItem('selectedDb', res.data.active_db);
        } else {
          localStorage.setItem('selectedDb', selectedDb);
        }

        if (tables.length > 0) {
          setSelectedTable(tables[0]);
        } else {
          setSelectedTable('');
          setCurrentTable(null);
          setLoadingList(false);
        }
      } catch (err) {
        console.error("Failed to load live tables:", err);
        setTablesList([]);
        setLoadingList(false);
      }
    };
    fetchTables();
  }, [selectedDb]);

  // Inspect selected table columns, indexes, and live row counts
  useEffect(() => {
    if (!selectedTable) return;
    const fetchTableDetails = async () => {
      try {
        setLoadingDetails(true);
        const url = `${API_BASE}/db/table/${selectedTable}?database=${selectedDb}`;
        const res = await axios.get(url);
        setCurrentTable(res.data);
      } catch (err) {
        console.error("Failed to inspect table details:", err);
        setCurrentTable({
          name: selectedTable,
          rows: 0,
          columns: []
        });
      } finally {
        setLoadingDetails(false);
        setLoadingList(false);
      }
    };
    fetchTableDetails();
  }, [selectedTable, selectedDb]);

  const filteredTables = tablesList.filter(name => 
    name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex flex-col gap-6 select-none">
      {/* Page Header */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-3">
          <div className="ds-brand-line" />
          <span className="ds-label-brand">Database Registry</span>
        </div>
        <h1 className="ds-heading-lg">Schema Registry Explorer</h1>
        <p className="ds-body">Browse ingested MySQL database tables, view primary key indexes, inspect foreign key relationships, and verify table constraints.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mt-4">
        {/* Left Column: Database & Tables Selector */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="ds-card p-4 flex flex-col gap-4">
            
            {/* Database Selector Dropdown */}
            <div className="flex flex-col gap-1.5">
              <label className="ds-field-label">Active MySQL Database</label>
              <select
                value={selectedDb}
                onChange={(e) => {
                  const dbName = e.target.value;
                  setSelectedDb(dbName);
                  localStorage.setItem('selectedDb', dbName);
                  setTablesList([]);
                  setSelectedTable('');
                  setCurrentTable(null);
                }}
                className="ds-select py-2 text-xs font-semibold focus:border-[var(--ds-brand)] bg-[var(--ds-surface-2)] cursor-pointer"
              >
                {databases.map(db => (
                  <option key={db} value={db}>{db}</option>
                ))}
              </select>
            </div>

            {/* Search Bar */}
            <div className="ds-input-wrap">
              <Search size={15} className="ds-input-icon !left-3 text-[var(--ds-text-faint)]" />
              <input
                type="text"
                placeholder="Search tables..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="ds-input py-2 pl-9 pr-4 text-xs font-semibold focus:border-[var(--ds-brand)] bg-[var(--ds-surface-2)]"
              />
            </div>

            {/* Tables list container */}
            <div className="flex flex-col gap-1 max-h-[350px] overflow-y-auto custom-scrollbar">
              {loadingList && tablesList.length === 0 ? (
                <div className="flex items-center justify-center p-8 gap-2 text-xs text-[var(--ds-text-faint)]">
                  <Loader2 className="animate-spin text-[var(--ds-brand)] w-3.5 h-3.5" /> Loading tables...
                </div>
              ) : filteredTables.map((tableName) => (
                <button
                  key={tableName}
                  onClick={() => setSelectedTable(tableName)}
                  className={`
                    w-full flex items-center justify-between px-4 py-3 rounded-lg text-xs font-semibold border transition-all text-left cursor-pointer
                    ${selectedTable === tableName 
                      ? 'bg-[var(--ds-nav-active-bg)] border-[var(--ds-border-strong)] text-[var(--ds-text)]' 
                      : 'border-transparent hover:bg-white/[0.01] text-[var(--ds-text-faint)] hover:text-[var(--ds-text)]'}
                  `}
                >
                  <span className="flex items-center gap-2 truncate pr-2">
                    <Table2 size={13} className={selectedTable === tableName ? 'text-[var(--ds-brand)]' : ''} />
                    <span className="truncate">{tableName}</span>
                  </span>
                </button>
              ))}
              
              {!loadingList && filteredTables.length === 0 && tablesList.length > 0 && (
                <span className="text-[11px] text-[var(--ds-text-faint)] italic p-2">No matching tables found.</span>
              )}

              {!loadingList && tablesList.length === 0 && (
                <span className="text-[11.5px] text-[var(--ds-text-faint)] italic p-2 text-center">
                  No tables inside this database.
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Column Specifications */}
        <div className="lg:col-span-8">
          {loadingList && tablesList.length === 0 ? (
            <div className="ds-card p-16 flex flex-col items-center justify-center">
              <Loader2 className="animate-spin text-[var(--ds-brand)] w-7 h-7 mb-2" />
              <span className="text-xs text-[var(--ds-text-muted)]">Inspecting schema structure...</span>
            </div>
          ) : tablesList.length === 0 ? (
            <div className="flex flex-col items-center justify-center p-12 ds-card text-center gap-4">
              <div className="w-12 h-12 rounded-full bg-[var(--ds-brand-dim)] flex items-center justify-center text-[var(--ds-brand)] border border-[var(--ds-brand-glow)]">
                <Table2 size={22} />
              </div>
              <div>
                <h3 className="ds-heading-sm">Empty Database Selected</h3>
                <p className="ds-body text-xs mt-2 max-w-sm">
                  This database has no active tables. Go to the Knowledge Portal to sync and ingest this schema dynamically!
                </p>
              </div>
              <Link 
                to="/ingestion"
                className="ds-btn-primary py-2.5 px-6 flex items-center gap-2 text-xs font-semibold"
              >
                Go to Knowledge Sync <ArrowRight size={14} />
              </Link>
            </div>
          ) : loadingDetails && !currentTable ? (
            <div className="ds-card p-12 flex flex-col items-center justify-center">
              <Loader2 className="animate-spin text-[var(--ds-brand)] w-6 h-6 mb-2" />
              <span className="text-xs text-[var(--ds-text-faint)]">Loading columns spec metadata...</span>
            </div>
          ) : currentTable ? (
            <div className="ds-card p-6 flex flex-col gap-6 relative">
              
              {/* Micro sync overlay indicator */}
              {loadingDetails && (
                <div className="absolute top-4 right-4 flex items-center gap-2 text-[10px] text-[var(--ds-text-faint)]">
                  <Loader2 className="animate-spin text-[var(--ds-brand)] w-3 h-3" /> Syncing...
                </div>
              )}

              <div className="flex items-center justify-between border-b border-[var(--ds-border)] pb-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-[var(--ds-brand-dim)] flex items-center justify-center text-[var(--ds-brand)] border border-[var(--ds-brand-glow)]">
                    <Table2 size={16} />
                  </div>
                  <div>
                    <h3 className="font-serif text-lg font-bold text-[var(--ds-text)]">{currentTable.name}</h3>
                    <p className="text-[10px] text-[var(--ds-text-faint)] uppercase tracking-wider font-semibold">
                      {currentTable.rows.toLocaleString()} Rows Introspected
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 bg-white/5 border border-[var(--ds-border)] rounded-full px-3 py-1 text-[10px] uppercase font-bold text-[var(--ds-text-muted)]">
                  <Eye size={12} className="text-[var(--ds-brand)]" /> Live Inspector
                </div>
              </div>

              {/* Table Details */}
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="ds-table-header">
                      <th>Column</th>
                      <th>Data Type</th>
                      <th>Index Rule</th>
                      <th>Constraint Specs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentTable.columns.map((col) => (
                      <tr key={col.name} className="ds-table-row">
                        <td className="font-semibold text-[var(--ds-text)]">{col.name}</td>
                        <td>
                          <code className="text-xs font-mono px-2 py-0.5 bg-white/5 border border-[var(--ds-border)] rounded text-[var(--ds-text-muted)]">
                            {col.type}
                          </code>
                        </td>
                        <td>
                          {col.primary ? (
                            <span className="ds-badge ds-badge-brand gap-1 text-[9px] py-0.5">
                              <Key size={9} /> Primary Key
                            </span>
                          ) : (
                            <span className="ds-badge ds-badge-neutral text-[9px] py-0.5">Standard</span>
                          )}
                        </td>
                        <td>
                          <span className={`text-xs font-mono font-medium ${col.extra && col.extra.includes('FK') ? 'text-[var(--ds-brand)]' : 'text-[var(--ds-text-faint)]'}`}>
                            {col.extra || 'NULLABLE: ' + (col.nullable ? 'YES' : 'NO')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
