import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  ArrowLeftRight,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Clock3,
  Copy,
  Database,
  FileText,
  FolderOpen,
  Image as ImageIcon,
  LayoutGrid,
  List,
  Moon,
  PencilLine,
  RefreshCw,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Tag,
  X,
} from 'lucide-react';

type NavKey = 'dataset' | 'analysis' | 'annotation' | 'evaluation' | 'conversions' | 'ai';
type TabKey = 'overview' | 'images' | 'labels' | 'classes' | 'splits';

interface DatasetImage {
  id: number;
  name: string;
  relative_path: string;
  split: string | null;
  width: number | null;
  height: number | null;
  boxes: number;
  label_exists: boolean;
  preview_url: string;
}

interface ClassRow {
  id: number;
  name: string;
  boxes: number;
  images: number;
  color: string;
}

interface SplitRow {
  name: string;
  images: number;
  boxes: number;
  ratio: number;
  class_counts: Record<string, number>;
}

interface AttributeRow {
  name: string;
  values: Record<string, number>;
}

interface Overview {
  root: string;
  layout: string;
  task: string;
  loaded_at: string;
  load_time_ms?: number;
  counts: {
    images: number;
    labels: number;
    boxes: number;
    classes: number;
    attributes: number;
    orphan_labels: number;
    empty_images: number;
  };
  classes: ClassRow[];
  attributes: AttributeRow[];
  splits: SplitRow[];
  images: DatasetImage[];
  operations: { name: string; time: string; detail: string }[];
}

interface ImagePage {
  total: number;
  offset: number;
  limit: number;
  items: DatasetImage[];
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const navItems: { key: NavKey; label: string; icon: typeof Database }[] = [
  { key: 'dataset', label: 'Dataset', icon: Database },
  { key: 'analysis', label: 'Analysis', icon: BarChart3 },
  { key: 'annotation', label: 'Annotation', icon: PencilLine },
  { key: 'evaluation', label: 'Evaluation', icon: ClipboardCheck },
  { key: 'conversions', label: 'Conversions', icon: ArrowLeftRight },
  { key: 'ai', label: 'AI tools', icon: Sparkles },
];

const tabs: { key: TabKey; label: string }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'images', label: 'Images' },
  { key: 'labels', label: 'Labels' },
  { key: 'classes', label: 'Classes' },
  { key: 'splits', label: 'Splits' },
];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json() as { detail?: string };
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status text when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function App() {
  const [activeNav, setActiveNav] = useState<NavKey>('dataset');
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [root, setRoot] = useState(() => localStorage.getItem('ydm.datasetRoot') || '');
  const [overview, setOverview] = useState<Overview | null>(null);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [search, setSearch] = useState('');
  const [splitFilter, setSplitFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'name' | 'boxes'>('name');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [darkMode, setDarkMode] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const selectedImage = useMemo(
    () => images.find((image) => image.id === selectedId) || images[0] || null,
    [images, selectedId],
  );

  const filteredImages = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return [...images].filter((image) => {
      const matchesSearch = !needle || image.name.toLowerCase().includes(needle) || image.relative_path.toLowerCase().includes(needle);
      const matchesSplit = splitFilter === 'all' || image.split === splitFilter;
      return matchesSearch && matchesSplit;
    }).sort((left, right) => sortBy === 'boxes' ? right.boxes - left.boxes || left.name.localeCompare(right.name) : left.name.localeCompare(right.name));
  }, [images, search, sortBy, splitFilter]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  async function loadImages() {
    const page = await request<ImagePage>('/api/dataset/images?limit=100');
    setImages(page.items);
    setSelectedId((current) => current ?? page.items[0]?.id ?? null);
  }

  async function handleLoad(event?: FormEvent) {
    event?.preventDefault();
    if (!root.trim()) {
      setError('Enter a dataset root or a dataset YAML path first.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const result = await request<Overview>('/api/dataset/load', {
        method: 'POST',
        body: JSON.stringify({ root: root.trim(), layout: 'auto', task: 'auto', workers: 8 }),
      });
      setOverview(result);
      localStorage.setItem('ydm.datasetRoot', root.trim());
      await loadImages();
      setActiveNav('dataset');
      setActiveTab('overview');
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Dataset load failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleRefresh() {
    if (!overview) return;
    setRefreshing(true);
    setError('');
    try {
      const result = await request<Overview>('/api/dataset/overview');
      setOverview(result);
      await loadImages();
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : 'Refresh failed.');
    } finally {
      setRefreshing(false);
    }
  }

  async function copyRoot() {
    if (!overview) return;
    try {
      await navigator.clipboard.writeText(overview.root);
    } catch {
      setError('The dataset path could not be copied by this browser.');
    }
  }

  function renderTabContent() {
    if (!overview) return <EmptyState onLoad={() => document.getElementById('dataset-root')?.focus()} />;
    if (activeTab === 'classes') {
      return <ClassesTable rows={overview.classes} />;
    }
    if (activeTab === 'splits') {
      return <SplitsTable rows={overview.splits} />;
    }
    if (activeTab === 'labels') {
      return (
        <div className="empty-tab">
          <FileText size={22} />
          <strong>{formatNumber(overview.counts.labels)} label files indexed</strong>
          <span>Use the Annotation and Analysis modules to inspect or update labels.</span>
        </div>
      );
    }
    return (
      <>
        <div className="toolbar-row">
          <div className="select-wrap">
            <SlidersHorizontal size={15} />
            <select value={splitFilter} onChange={(event) => setSplitFilter(event.target.value)} aria-label="Filter split">
              <option value="all">All images</option>
              <option value="train">Train</option>
              <option value="val">Val</option>
              <option value="test">Test</option>
            </select>
          </div>
          <div className="select-wrap">
            <select aria-label="Sort images" value={sortBy} onChange={(event) => setSortBy(event.target.value as 'name' | 'boxes')}>
              <option value="name">Sort: Name (A–Z)</option>
              <option value="boxes">Sort: Box count</option>
            </select>
          </div>
          <div className="view-toggle" aria-label="View mode">
            <button className={viewMode === 'grid' ? 'selected' : ''} onClick={() => setViewMode('grid')} title="Grid view"><LayoutGrid size={15} /></button>
            <button className={viewMode === 'list' ? 'selected' : ''} onClick={() => setViewMode('list')} title="List view"><List size={15} /></button>
          </div>
          <span className="toolbar-count">{formatNumber(filteredImages.length)} images</span>
          <div className="pager"><button disabled><ChevronLeft size={15} /></button><span>1 / {Math.max(1, Math.ceil(filteredImages.length / 100))}</span><button disabled><ChevronRight size={15} /></button></div>
        </div>
        <ImageBrowser images={filteredImages} selectedId={selectedImage?.id ?? null} onSelect={setSelectedId} viewMode={viewMode} />
        {activeTab === 'overview' && selectedImage && <PreviewPanel image={selectedImage} overview={overview} images={filteredImages} onSelect={setSelectedId} />}
      </>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Database size={19} strokeWidth={2.4} /></div>
          <div className="brand-name">YDM</div>
          <div className="brand-subtitle">YOLO Data Manager</div>
        </div>
        <label className="global-search">
          <Search size={16} />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search images, classes, or files..." />
          {search && <button onClick={() => setSearch('')} aria-label="Clear search"><X size={14} /></button>}
        </label>
        <div className="top-actions">
          <button className="icon-button" onClick={() => setDarkMode((value) => !value)} title="Toggle theme">{darkMode ? <Sun size={18} /> : <Moon size={18} />}</button>
          <div className="settings-wrap">
            <button className={`icon-button ${settingsOpen ? 'active' : ''}`} onClick={() => setSettingsOpen((value) => !value)} title="Settings"><Settings size={18} /></button>
            {settingsOpen && <div className="settings-popover"><strong>Standalone workspace</strong><span>API · 127.0.0.1:8091</span><span>Frontend · 127.0.0.1:5174</span><small>Runtime options are configured when YDM starts.</small></div>}
          </div>
        </div>
      </header>

      <div className="workspace-frame">
        <aside className="sidebar">
          <div className="sidebar-label">Workspace</div>
          <nav>
            {navItems.map(({ key, label, icon: Icon }) => (
              <button key={key} className={`nav-item ${activeNav === key ? 'active' : ''}`} onClick={() => setActiveNav(key)}>
                <Icon size={18} strokeWidth={activeNav === key ? 2.3 : 1.8} /><span>{label}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer"><div className="status-dot" /> <span>Local workspace</span><span className="version-label">v1.0.0</span></div>
        </aside>

        <main className="main-area">
          <div className="content-header">
            <div className="breadcrumbs"><span>Dataset</span><ChevronRight size={14} /><strong>{activeNav === 'dataset' ? 'Overview' : navItems.find((item) => item.key === activeNav)?.label}</strong></div>
            <div className="content-heading-row">
              <div><h1>{activeNav === 'dataset' ? 'Overview' : navItems.find((item) => item.key === activeNav)?.label}</h1><p>{activeNav === 'dataset' ? 'Load a YOLO dataset to inspect its images, labels, classes, and splits.' : 'This workspace is ready for the next YDM module.'}</p></div>
              <button className="primary-button" onClick={() => document.getElementById('dataset-root')?.focus()}><FolderOpen size={16} /> Load dataset</button>
            </div>
          </div>

          {error && <div className="error-banner"><AlertCircle size={17} /><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}

          {activeNav !== 'dataset' ? (
            <ModulePlaceholder label={navItems.find((item) => item.key === activeNav)?.label || 'Module'} onReturn={() => setActiveNav('dataset')} />
          ) : (
            <>
              <form className="load-bar" onSubmit={handleLoad}>
                <div className="path-field"><FolderOpen size={16} /><input id="dataset-root" value={root} onChange={(event) => setRoot(event.target.value)} placeholder="Dataset root or data.yaml path" /><button type="button" className="path-clear" onClick={() => setRoot('')} title="Clear path"><X size={14} /></button></div>
                <button className="primary-button load-submit" type="submit" disabled={loading}>{loading ? <RefreshCw className="spin" size={16} /> : <Database size={16} />}{loading ? 'Loading…' : 'Load dataset'}</button>
              </form>

              {overview && (
                <div className="tabbar">
                  {tabs.map((tab) => <button key={tab.key} className={activeTab === tab.key ? 'active' : ''} onClick={() => setActiveTab(tab.key)}>{tab.label}</button>)}
                </div>
              )}

              <div className="workspace-content">{renderTabContent()}</div>
            </>
          )}
        </main>

        <aside className="inspector">
          <Inspector overview={overview} onCopy={copyRoot} onRefresh={handleRefresh} refreshing={refreshing} />
        </aside>
      </div>
    </div>
  );
}

function EmptyState({ onLoad }: { onLoad: () => void }) {
  return <div className="empty-state"><div className="empty-icon"><Database size={28} /></div><h2>Load a dataset to begin</h2><p>Point YDM at a YOLO dataset root or its data YAML. The workspace will index images, labels, classes, attributes, and split files locally.</p><button className="primary-button" onClick={onLoad}><FolderOpen size={16} /> Choose dataset path</button></div>;
}

function ModulePlaceholder({ label, onReturn }: { label: string; onReturn: () => void }) {
  return <div className="module-placeholder"><div className="empty-icon"><Sparkles size={26} /></div><h2>{label} is ready to connect</h2><p>The standalone YDM shell is in place. This module will reuse the same workspace, logs, and dataset session as its operation API is exposed.</p><button className="secondary-button" onClick={onReturn}>Return to Dataset</button></div>;
}

function ImageBrowser({ images, selectedId, onSelect, viewMode }: { images: DatasetImage[]; selectedId: number | null; onSelect: (id: number) => void; viewMode: 'grid' | 'list' }) {
  if (!images.length) return <div className="no-results"><Search size={18} /> No images match the current filter.</div>;
  return <div className={viewMode === 'grid' ? 'image-strip' : 'image-list'}>{images.slice(0, 12).map((image) => viewMode === 'grid' ? <button key={image.id} className={`image-card ${selectedId === image.id ? 'selected' : ''}`} onClick={() => onSelect(image.id)}><img src={`${API_BASE}${image.preview_url}`} alt={image.name} loading="lazy" /><span className="image-card-name">{image.name}</span><span className="image-card-meta">{image.boxes} {image.boxes === 1 ? 'box' : 'boxes'}{image.split ? ` · ${image.split}` : ''}</span></button> : <button key={image.id} className={`image-list-row ${selectedId === image.id ? 'selected' : ''}`} onClick={() => onSelect(image.id)}><ImageIcon size={16} /><span>{image.name}</span><span>{image.split || '—'}</span><span>{image.boxes}</span></button>)}</div>;
}

function PreviewPanel({ image, overview, images, onSelect }: { image: DatasetImage; overview: Overview; images: DatasetImage[]; onSelect: (id: number) => void }) {
  const currentIndex = images.findIndex((item) => item.id === image.id);
  const previous = currentIndex > 0 ? images[currentIndex - 1] : null;
  const next = currentIndex >= 0 && currentIndex < images.length - 1 ? images[currentIndex + 1] : null;
  return <section className="preview-panel"><div className="preview-header"><div><strong>{image.name}</strong><span>{image.width && image.height ? `${image.width} × ${image.height}` : 'image dimensions unavailable'}</span></div><div className="preview-actions"><button className="icon-button small" title="Previous image" disabled={!previous} onClick={() => previous && onSelect(previous.id)}><ChevronLeft size={15} /></button><span>Image {image.id + 1} / {overview.counts.images}</span><button className="icon-button small" title="Next image" disabled={!next} onClick={() => next && onSelect(next.id)}><ChevronRight size={15} /></button></div></div><div className="preview-canvas"><img src={`${API_BASE}${image.preview_url}`} alt={`Annotated preview of ${image.name}`} /></div></section>;
}

function Inspector({ overview, onCopy, onRefresh, refreshing }: { overview: Overview | null; onCopy: () => void; onRefresh: () => void; refreshing: boolean }) {
  if (!overview) return <div className="inspector-empty"><Database size={22} /><strong>No dataset loaded</strong><span>Dataset details and recent operations will appear here.</span></div>;
  return <div className="inspector-stack"><section className="inspector-card"><div className="card-title"><FolderOpen size={16} /><strong>Dataset</strong><button className="icon-button small" onClick={onRefresh} title="Refresh dataset">{refreshing ? <RefreshCw className="spin" size={15} /> : <RefreshCw size={15} />}</button></div><div className="root-box"><span>{overview.root}</span><button onClick={onCopy} title="Copy dataset path"><Copy size={14} /></button></div><div className="summary-list"><SummaryLine label="Images" value={formatNumber(overview.counts.images)} /><SummaryLine label="Labels" value={formatNumber(overview.counts.labels)} /><SummaryLine label="Boxes" value={formatNumber(overview.counts.boxes)} /><SummaryLine label="Classes" value={formatNumber(overview.counts.classes)} /></div><div className="divider" /><div className="split-summary">{overview.splits.filter((split) => split.name !== 'unassigned').map((split) => <div key={split.name}><span>{split.name[0].toUpperCase() + split.name.slice(1)}</span><strong>{formatNumber(split.images)} <em>({Math.round(split.ratio * 100)}%)</em></strong></div>)}</div><div className="dataset-meta"><span>{overview.layout} layout</span><span>{overview.task} task</span></div></section><section className="inspector-card"><div className="card-title"><Tag size={16} /><strong>Classes</strong><span className="card-count">{overview.classes.length}</span></div><div className="class-list">{overview.classes.slice(0, 8).map((row) => <div className="class-line" key={row.id}><span className="class-swatch" style={{ background: row.color }} /><span>{row.name}</span><strong>{formatNumber(row.boxes)}</strong></div>)}{overview.classes.length > 8 && <span className="muted-note">+ {overview.classes.length - 8} more classes</span>}</div></section><section className="inspector-card"><div className="card-title"><Clock3 size={16} /><strong>Recent operations</strong></div><div className="operation-list">{overview.operations.map((operation, index) => <div className="operation-line" key={`${operation.name}-${index}`}><FileText size={14} /><div><strong>{operation.name}</strong><span>{operation.detail}</span></div><time>{formatTime(operation.time)}</time></div>)}</div></section></div>;
}

function SummaryLine({ label, value }: { label: string; value: string }) { return <div className="summary-line"><span>{label}</span><strong>{value}</strong></div>; }

function ClassesTable({ rows }: { rows: ClassRow[] }) { return <div className="data-table-wrap"><table className="data-table"><thead><tr><th>ID</th><th>Class</th><th>Boxes</th><th>Images</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id}><td>{row.id}</td><td><span className="class-swatch" style={{ background: row.color }} />{row.name}</td><td>{formatNumber(row.boxes)}</td><td>{formatNumber(row.images)}</td></tr>)}</tbody></table></div>; }

function SplitsTable({ rows }: { rows: SplitRow[] }) { return <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Split</th><th>Images</th><th>Boxes</th><th>Ratio</th></tr></thead><tbody>{rows.map((row) => <tr key={row.name}><td>{row.name}</td><td>{formatNumber(row.images)}</td><td>{formatNumber(row.boxes)}</td><td>{Math.round(row.ratio * 100)}%</td></tr>)}</tbody></table></div>; }

export default App;
