import { FormEvent, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
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
  Languages,
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

const navItems: { key: NavKey; labelKey: string; icon: typeof Database }[] = [
  { key: 'dataset', labelKey: 'nav.dataset', icon: Database },
  { key: 'analysis', labelKey: 'nav.analysis', icon: BarChart3 },
  { key: 'annotation', labelKey: 'nav.annotation', icon: PencilLine },
  { key: 'evaluation', labelKey: 'nav.evaluation', icon: ClipboardCheck },
  { key: 'conversions', labelKey: 'nav.conversions', icon: ArrowLeftRight },
  { key: 'ai', labelKey: 'nav.aiTools', icon: Sparkles },
];

const tabs: { key: TabKey; labelKey: string }[] = [
  { key: 'overview', labelKey: 'tabs.overview' },
  { key: 'images', labelKey: 'tabs.images' },
  { key: 'labels', labelKey: 'tabs.labels' },
  { key: 'classes', labelKey: 'tabs.classes' },
  { key: 'splits', labelKey: 'tabs.splits' },
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

function formatNumber(value: number, language = 'en'): string {
  return new Intl.NumberFormat(language.startsWith('zh') ? 'zh-CN' : 'en-US').format(value);
}

function formatTime(value: string, language = 'en'): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleTimeString(language.startsWith('zh') ? 'zh-CN' : 'en-US', { hour: '2-digit', minute: '2-digit' });
}

function App() {
  const { t, i18n } = useTranslation();
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
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';

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

  useEffect(() => {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en-US';
  }, [language]);

  async function loadImages() {
    const page = await request<ImagePage>('/api/dataset/images?limit=100');
    setImages(page.items);
    setSelectedId((current) => current ?? page.items[0]?.id ?? null);
  }

  async function handleLoad(event?: FormEvent) {
    event?.preventDefault();
    if (!root.trim()) {
      setError(t('dataset.enterPath'));
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
      setError(loadError instanceof Error ? loadError.message : t('dataset.loadFailed'));
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
      setError(refreshError instanceof Error ? refreshError.message : t('dataset.refreshFailed'));
    } finally {
      setRefreshing(false);
    }
  }

  async function copyRoot() {
    if (!overview) return;
    try {
      await navigator.clipboard.writeText(overview.root);
    } catch {
      setError(t('dataset.copyPathFailed'));
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
          <strong>{t('dataset.labelsIndexed', { count: formatNumber(overview.counts.labels, language) })}</strong>
          <span>{t('dataset.labelsHint')}</span>
        </div>
      );
    }
    return (
      <>
        <div className="toolbar-row">
          <div className="select-wrap">
            <SlidersHorizontal size={15} />
            <select value={splitFilter} onChange={(event) => setSplitFilter(event.target.value)} aria-label={t('dataset.filterSplit')}>
              <option value="all">{t('dataset.allImages')}</option>
              <option value="train">{t('dataset.train')}</option>
              <option value="val">{t('dataset.val')}</option>
              <option value="test">{t('dataset.test')}</option>
            </select>
          </div>
          <div className="select-wrap">
            <select aria-label={t('dataset.sortImages')} value={sortBy} onChange={(event) => setSortBy(event.target.value as 'name' | 'boxes')}>
              <option value="name">{t('dataset.sortName')}</option>
              <option value="boxes">{t('dataset.sortBoxes')}</option>
            </select>
          </div>
          <div className="view-toggle" aria-label={t('dataset.viewMode')}>
            <button className={viewMode === 'grid' ? 'selected' : ''} onClick={() => setViewMode('grid')} title={t('common.gridView')}><LayoutGrid size={15} /></button>
            <button className={viewMode === 'list' ? 'selected' : ''} onClick={() => setViewMode('list')} title={t('common.listView')}><List size={15} /></button>
          </div>
          <span className="toolbar-count">{t('dataset.imageCount', { count: formatNumber(filteredImages.length, language) })}</span>
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
          <div className="brand-subtitle">{t('brand.subtitle')}</div>
        </div>
        <label className="global-search">
          <Search size={16} />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t('search.placeholder')} />
          {search && <button onClick={() => setSearch('')} aria-label={t('search.clear')}><X size={14} /></button>}
        </label>
        <div className="top-actions">
          <button
            className="icon-button language-button"
            onClick={() => void i18n.changeLanguage(language === 'zh' ? 'en' : 'zh')}
            title={t(language === 'zh' ? 'language.switchToEnglish' : 'language.switchToChinese')}
            aria-label={t(language === 'zh' ? 'language.switchToEnglish' : 'language.switchToChinese')}
          >
            <Languages size={16} />
            <span>{language === 'zh' ? 'EN' : '中'}</span>
          </button>
          <button className="icon-button" onClick={() => setDarkMode((value) => !value)} title={t('common.toggleTheme')} aria-label={t('common.toggleTheme')}>{darkMode ? <Sun size={18} /> : <Moon size={18} />}</button>
          <div className="settings-wrap">
            <button className={`icon-button ${settingsOpen ? 'active' : ''}`} onClick={() => setSettingsOpen((value) => !value)} title={t('common.settings')} aria-label={t('common.settings')}><Settings size={18} /></button>
            {settingsOpen && <div className="settings-popover"><strong>{t('settings.standaloneWorkspace')}</strong><span>{t('settings.api')}</span><span>{t('settings.frontend')}</span><small>{t('settings.runtimeHint')}</small></div>}
          </div>
        </div>
      </header>

      <div className="workspace-frame">
        <aside className="sidebar">
          <div className="sidebar-label">{t('nav.workspace')}</div>
          <nav>
            {navItems.map(({ key, labelKey, icon: Icon }) => (
              <button key={key} className={`nav-item ${activeNav === key ? 'active' : ''}`} onClick={() => setActiveNav(key)}>
                <Icon size={18} strokeWidth={activeNav === key ? 2.3 : 1.8} /><span>{t(labelKey)}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer"><div className="status-dot" /> <span>{t('nav.workspace')}</span><span className="version-label">v1.0.1</span></div>
        </aside>

        <main className="main-area">
          <div className="content-header">
            <div className="breadcrumbs"><span>{t('nav.dataset')}</span><ChevronRight size={14} /><strong>{activeNav === 'dataset' ? t('tabs.overview') : t(navItems.find((item) => item.key === activeNav)?.labelKey || 'nav.module')}</strong></div>
            <div className="content-heading-row">
              <div><h1>{activeNav === 'dataset' ? t('tabs.overview') : t(navItems.find((item) => item.key === activeNav)?.labelKey || 'nav.module')}</h1><p>{activeNav === 'dataset' ? t('dataset.description') : t('module.description')}</p></div>
            </div>
          </div>

          {error && <div className="error-banner"><AlertCircle size={17} /><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}

          {activeNav !== 'dataset' ? (
            <ModulePlaceholder label={t(navItems.find((item) => item.key === activeNav)?.labelKey || 'nav.module')} onReturn={() => setActiveNav('dataset')} />
          ) : (
            <>
              <form className="load-bar" onSubmit={handleLoad}>
                <div className="path-field"><FolderOpen size={16} /><input id="dataset-root" value={root} onChange={(event) => setRoot(event.target.value)} placeholder={t('dataset.pathPlaceholder')} /><button type="button" className="path-clear" onClick={() => setRoot('')} title={t('common.clear')} aria-label={t('common.clear')}><X size={14} /></button></div>
                <button className="primary-button load-submit" type="submit" disabled={loading}>{loading ? <RefreshCw className="spin" size={16} /> : <Database size={16} />}{loading ? t('dataset.loading') : t('common.loadDataset')}</button>
              </form>

              {overview && (
                <div className="tabbar">
                  {tabs.map((tab) => <button key={tab.key} className={activeTab === tab.key ? 'active' : ''} onClick={() => setActiveTab(tab.key)}>{t(tab.labelKey)}</button>)}
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
  const { t } = useTranslation();
  return <div className="empty-state"><div className="empty-icon"><Database size={28} /></div><h2>{t('empty.title')}</h2><p>{t('empty.description')}</p><button className="primary-button" onClick={onLoad}><FolderOpen size={16} /> {t('common.chooseDatasetPath')}</button></div>;
}

function ModulePlaceholder({ label, onReturn }: { label: string; onReturn: () => void }) {
  const { t } = useTranslation();
  return <div className="module-placeholder"><div className="empty-icon"><Sparkles size={26} /></div><h2>{t('module.ready', { label })}</h2><p>{t('module.description')}</p><button className="secondary-button" onClick={onReturn}>{t('common.returnToDataset')}</button></div>;
}

function ImageBrowser({ images, selectedId, onSelect, viewMode }: { images: DatasetImage[]; selectedId: number | null; onSelect: (id: number) => void; viewMode: 'grid' | 'list' }) {
  const { t, i18n } = useTranslation();
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  if (!images.length) return <div className="no-results"><Search size={18} /> {t('result.noImages')}</div>;
  return <div className={viewMode === 'grid' ? 'image-strip' : 'image-list'}>{images.slice(0, 12).map((image) => viewMode === 'grid' ? <button key={image.id} className={`image-card ${selectedId === image.id ? 'selected' : ''}`} onClick={() => onSelect(image.id)}><img src={`${API_BASE}${image.preview_url}`} alt={image.name} loading="lazy" /><span className="image-card-name">{image.name}</span><span className="image-card-meta">{formatNumber(image.boxes, language)} {image.boxes === 1 ? t('dataset.boxSingular') : t('dataset.boxPlural')}{image.split ? ` · ${image.split}` : ''}</span></button> : <button key={image.id} className={`image-list-row ${selectedId === image.id ? 'selected' : ''}`} onClick={() => onSelect(image.id)}><ImageIcon size={16} /><span>{image.name}</span><span>{image.split || '—'}</span><span>{formatNumber(image.boxes, language)}</span></button>)}</div>;
}

function PreviewPanel({ image, overview, images, onSelect }: { image: DatasetImage; overview: Overview; images: DatasetImage[]; onSelect: (id: number) => void }) {
  const { t, i18n } = useTranslation();
  const currentIndex = images.findIndex((item) => item.id === image.id);
  const previous = currentIndex > 0 ? images[currentIndex - 1] : null;
  const next = currentIndex >= 0 && currentIndex < images.length - 1 ? images[currentIndex + 1] : null;
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  return <section className="preview-panel"><div className="preview-header"><div><strong>{image.name}</strong><span>{image.width && image.height ? `${image.width} × ${image.height}` : t('dataset.imageDimensionsUnavailable')}</span></div><div className="preview-actions"><button className="icon-button small" title={t('common.previousImage')} aria-label={t('common.previousImage')} disabled={!previous} onClick={() => previous && onSelect(previous.id)}><ChevronLeft size={15} /></button><span>{t('dataset.imagePage', { current: formatNumber(image.id + 1, language), total: formatNumber(overview.counts.images, language) })}</span><button className="icon-button small" title={t('common.nextImage')} aria-label={t('common.nextImage')} disabled={!next} onClick={() => next && onSelect(next.id)}><ChevronRight size={15} /></button></div></div><div className="preview-canvas"><img src={`${API_BASE}${image.preview_url}`} alt={`${t('tabs.images')} ${image.name}`} /></div></section>;
}

function Inspector({ overview, onCopy, onRefresh, refreshing }: { overview: Overview | null; onCopy: () => void; onRefresh: () => void; refreshing: boolean }) {
  const { t, i18n } = useTranslation();
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  if (!overview) return <div className="inspector-empty"><Database size={22} /><strong>{t('inspector.noDataset')}</strong><span>{t('inspector.description')}</span></div>;
  return <div className="inspector-stack"><section className="inspector-card"><div className="card-title"><FolderOpen size={16} /><strong>{t('inspector.dataset')}</strong><button className="icon-button small" onClick={onRefresh} title={t('common.refreshDataset')} aria-label={t('common.refreshDataset')}>{refreshing ? <RefreshCw className="spin" size={15} /> : <RefreshCw size={15} />}</button></div><div className="root-box"><span>{overview.root}</span><button onClick={onCopy} title={t('common.copyDatasetPath')} aria-label={t('common.copyDatasetPath')}><Copy size={14} /></button></div><div className="summary-list"><SummaryLine label={t('inspector.images')} value={formatNumber(overview.counts.images, language)} /><SummaryLine label={t('inspector.labels')} value={formatNumber(overview.counts.labels, language)} /><SummaryLine label={t('inspector.boxes')} value={formatNumber(overview.counts.boxes, language)} /><SummaryLine label={t('inspector.classes')} value={formatNumber(overview.counts.classes, language)} /></div><div className="divider" /><div className="split-summary">{overview.splits.filter((split) => split.name !== 'unassigned').map((split) => <div key={split.name}><span>{split.name[0].toUpperCase() + split.name.slice(1)}</span><strong>{formatNumber(split.images, language)} <em>({Math.round(split.ratio * 100)}%)</em></strong></div>)}</div><div className="dataset-meta"><span>{t('dataset.layout', { value: overview.layout })}</span><span>{t('dataset.task', { value: overview.task })}</span></div></section><section className="inspector-card"><div className="card-title"><Tag size={16} /><strong>{t('inspector.classes')}</strong><span className="card-count">{overview.classes.length}</span></div><div className="class-list">{overview.classes.slice(0, 8).map((row) => <div className="class-line" key={row.id}><span className="class-swatch" style={{ background: row.color }} /><span>{row.name}</span><strong>{formatNumber(row.boxes, language)}</strong></div>)}{overview.classes.length > 8 && <span className="muted-note">{t('inspector.moreClasses', { count: overview.classes.length - 8 })}</span>}</div></section><section className="inspector-card"><div className="card-title"><Clock3 size={16} /><strong>{t('inspector.recentOperations')}</strong></div><div className="operation-list">{overview.operations.map((operation, index) => <div className="operation-line" key={`${operation.name}-${index}`}><FileText size={14} /><div><strong>{operation.name}</strong><span>{operation.detail}</span></div><time>{formatTime(operation.time, language)}</time></div>)}</div></section></div>;
}

function SummaryLine({ label, value }: { label: string; value: string }) { return <div className="summary-line"><span>{label}</span><strong>{value}</strong></div>; }

function ClassesTable({ rows }: { rows: ClassRow[] }) {
  const { t, i18n } = useTranslation();
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  return <div className="data-table-wrap"><table className="data-table"><thead><tr><th>{t('table.id')}</th><th>{t('table.class')}</th><th>{t('table.boxes')}</th><th>{t('table.images')}</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id}><td>{row.id}</td><td><span className="class-swatch" style={{ background: row.color }} />{row.name}</td><td>{formatNumber(row.boxes, language)}</td><td>{formatNumber(row.images, language)}</td></tr>)}</tbody></table></div>;
}

function SplitsTable({ rows }: { rows: SplitRow[] }) {
  const { t, i18n } = useTranslation();
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  return <div className="data-table-wrap"><table className="data-table"><thead><tr><th>{t('table.split')}</th><th>{t('table.images')}</th><th>{t('table.boxes')}</th><th>{t('table.ratio')}</th></tr></thead><tbody>{rows.map((row) => <tr key={row.name}><td>{row.name}</td><td>{formatNumber(row.images, language)}</td><td>{formatNumber(row.boxes, language)}</td><td>{Math.round(row.ratio * 100)}%</td></tr>)}</tbody></table></div>;
}

export default App;
