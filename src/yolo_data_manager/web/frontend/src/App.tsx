import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertCircle,
  ArrowLeftRight,
  Bot,
  BarChart3,
  Check,
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
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Tag,
  Terminal,
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

interface VLMStatus {
  configured: boolean;
  provider: string;
  model: string;
  base_url: string | null;
  dotenv_path: string | null;
  error?: string | null;
}

interface AssistantPlan {
  method: string;
  arguments: Record<string, unknown>;
  explanation: string;
  requires_confirmation: boolean;
}

interface AssistantResponse {
  provider: string;
  model: string;
  plan: AssistantPlan;
  executed: boolean;
  result: unknown;
}

interface AssistantMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  response?: AssistantResponse;
}

interface RuntimeLog {
  id: number;
  time: string;
  level: 'command' | 'info' | 'success' | 'warning' | 'error' | string;
  message: string;
  progress?: RuntimeProgress;
}

interface RuntimeProgress {
  stage: string;
  current: number;
  total: number | null;
  percent: number | null;
  elapsed_seconds: number;
  rate: number | null;
  eta_seconds: number | null;
  unit: string;
  done: boolean;
}

interface RuntimeLogPage {
  items: RuntimeLog[];
  next_id: number;
  latest_id: number;
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

function formatDuration(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value < 60) return `${Math.max(0, Math.round(value))}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
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
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantStatus, setAssistantStatus] = useState<VLMStatus | null>(null);
  const [assistantStatusLoading, setAssistantStatusLoading] = useState(false);
  const [assistantInput, setAssistantInput] = useState('');
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([]);
  const [assistantPending, setAssistantPending] = useState<AssistantResponse | null>(null);
  const [assistantBusy, setAssistantBusy] = useState(false);
  const [runLogs, setRunLogs] = useState<RuntimeLog[]>([]);
  const [bottomTab, setBottomTab] = useState<'details' | 'logs'>('logs');
  const logCursorRef = useRef(0);
  const latestLogIdRef = useRef(0);
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

  useEffect(() => {
    let stopped = false;
    let inFlight = false;

    const pollRuntimeLogs = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const page = await request<RuntimeLogPage>(`/api/logs?after=${logCursorRef.current}&limit=200`);
        if (stopped) return;
        if (page.items.length) {
          setRunLogs((current) => [...current, ...page.items].slice(-400));
        }
        logCursorRef.current = page.next_id;
        latestLogIdRef.current = page.latest_id;
      } catch {
        // The terminal is diagnostic UI; a temporary API restart should not
        // replace the main workspace error state.
      } finally {
        inFlight = false;
      }
    };

    void pollRuntimeLogs();
    const timer = window.setInterval(() => void pollRuntimeLogs(), 1200);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);

  function clearRunLogs() {
    setRunLogs([]);
    logCursorRef.current = latestLogIdRef.current;
  }

  useEffect(() => {
    if (!assistantOpen || assistantStatus || assistantStatusLoading) return;
    setAssistantStatusLoading(true);
    void request<VLMStatus>('/api/vlm/status')
      .then(setAssistantStatus)
      .catch((statusError) => setAssistantStatus({
        configured: false,
        provider: 'qwen',
        model: '',
        base_url: null,
        dotenv_path: null,
        error: statusError instanceof Error ? statusError.message : String(statusError),
      }))
      .finally(() => setAssistantStatusLoading(false));
  }, [assistantOpen, assistantStatus, assistantStatusLoading]);

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
    setBottomTab('logs');
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

  function appendAssistantMessage(message: Omit<AssistantMessage, 'id'>) {
    setAssistantMessages((current) => [...current, { ...message, id: Date.now() + current.length }]);
  }

  async function handleAssistantSend(event?: FormEvent) {
    event?.preventDefault();
    const intent = assistantInput.trim();
    if (!intent || assistantBusy || !overview) return;
    appendAssistantMessage({ role: 'user', content: intent });
    setAssistantInput('');
    setAssistantPending(null);
    setAssistantBusy(true);
    try {
      const response = await request<AssistantResponse>('/api/vlm/assistant', {
        method: 'POST',
        body: JSON.stringify({ intent, execute: false, confirm: false }),
      });
      appendAssistantMessage({
        role: 'assistant',
        content: response.plan.explanation || t('assistant.planReady'),
        response,
      });
      setAssistantPending(response);
    } catch (assistantError) {
      appendAssistantMessage({
        role: 'assistant',
        content: assistantError instanceof Error ? assistantError.message : t('assistant.error'),
      });
    } finally {
      setAssistantBusy(false);
    }
  }

  async function handleAssistantConfirm() {
    if (!assistantPending || assistantBusy || !overview) return;
    setAssistantBusy(true);
    try {
      const response = await request<AssistantResponse>('/api/vlm/assistant', {
        method: 'POST',
        body: JSON.stringify({ plan: assistantPending.plan, execute: true, confirm: true }),
      });
      appendAssistantMessage({
        role: 'assistant',
        content: response.executed ? t('assistant.executed') : t('assistant.planReady'),
        response,
      });
      setAssistantPending(null);
      if (response.executed) await handleRefresh();
    } catch (assistantError) {
      appendAssistantMessage({
        role: 'assistant',
        content: assistantError instanceof Error ? assistantError.message : t('assistant.error'),
      });
    } finally {
      setAssistantBusy(false);
    }
  }

  function handleAssistantCancel() {
    setAssistantPending(null);
    appendAssistantMessage({ role: 'assistant', content: t('assistant.cancelled') });
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
          <button
            className={`icon-button assistant-toggle ${assistantOpen ? 'active' : ''}`}
            onClick={() => setAssistantOpen((value) => !value)}
            title={t(assistantOpen ? 'assistant.close' : 'assistant.open')}
            aria-label={t(assistantOpen ? 'assistant.close' : 'assistant.open')}
          >
            <Bot size={18} />
          </button>
          <button className="icon-button" onClick={() => setDarkMode((value) => !value)} title={t('common.toggleTheme')} aria-label={t('common.toggleTheme')}>{darkMode ? <Sun size={18} /> : <Moon size={18} />}</button>
          <div className="settings-wrap">
            <button className={`icon-button ${settingsOpen ? 'active' : ''}`} onClick={() => setSettingsOpen((value) => !value)} title={t('common.settings')} aria-label={t('common.settings')}><Settings size={18} /></button>
            {settingsOpen && <div className="settings-popover"><strong>{t('settings.standaloneWorkspace')}</strong><span>{t('settings.api')}</span><span>{t('settings.frontend')}</span><small>{t('settings.runtimeHint')}</small></div>}
          </div>
        </div>
      </header>

      <div className={`workspace-frame ${assistantOpen ? 'assistant-workspace' : ''}`}>
        <aside className="sidebar">
          <div className="sidebar-label">{t('nav.workspace')}</div>
          <nav>
            {navItems.map(({ key, labelKey, icon: Icon }) => (
              <button key={key} className={`nav-item ${activeNav === key ? 'active' : ''}`} onClick={() => setActiveNav(key)}>
                <Icon size={18} strokeWidth={activeNav === key ? 2.3 : 1.8} /><span>{t(labelKey)}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-footer"><div className="status-dot" /> <span>{t('nav.workspace')}</span><span className="version-label">v1.2.5</span></div>
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

          <BottomPanel
            activeTab={bottomTab}
            onTabChange={setBottomTab}
            overview={overview}
            onCopy={copyRoot}
            onRefresh={handleRefresh}
            refreshing={refreshing}
            logs={runLogs}
            onClearLogs={clearRunLogs}
          />
        </main>

        {assistantOpen && (
          <aside className="right-rail assistant-open">
            <AssistantDrawer
              overview={overview}
              selectedImage={selectedImage}
              status={assistantStatus}
              statusLoading={assistantStatusLoading}
              input={assistantInput}
              messages={assistantMessages}
              pendingPlan={assistantPending}
              busy={assistantBusy}
              onClose={() => setAssistantOpen(false)}
              onInputChange={setAssistantInput}
              onSubmit={handleAssistantSend}
              onConfirm={handleAssistantConfirm}
              onCancel={handleAssistantCancel}
              onQuickPrompt={setAssistantInput}
            />
          </aside>
        )}
      </div>
    </div>
  );
}

function BottomPanel({
  activeTab,
  onTabChange,
  overview,
  onCopy,
  onRefresh,
  refreshing,
  logs,
  onClearLogs,
}: {
  activeTab: 'details' | 'logs';
  onTabChange: (tab: 'details' | 'logs') => void;
  overview: Overview | null;
  onCopy: () => void;
  onRefresh: () => void;
  refreshing: boolean;
  logs: RuntimeLog[];
  onClearLogs: () => void;
}) {
  const { t } = useTranslation();
  return (
    <section className="bottom-panel">
      <div className="bottom-panel-tabs" role="tablist" aria-label={t('bottomPanel.title')}>
        <button
          className={activeTab === 'details' ? 'active' : ''}
          role="tab"
          aria-selected={activeTab === 'details'}
          onClick={() => onTabChange('details')}
        >
          <Database size={14} />
          {t('inspector.details')}
        </button>
        <button
          className={activeTab === 'logs' ? 'active' : ''}
          role="tab"
          aria-selected={activeTab === 'logs'}
          onClick={() => onTabChange('logs')}
        >
          <Terminal size={14} />
          {t('terminal.title')}
          <span className="bottom-panel-count">{logs.length}</span>
        </button>
      </div>
      {activeTab === 'details' ? (
        <div className="bottom-details" role="tabpanel">
          <Inspector overview={overview} onCopy={onCopy} onRefresh={onRefresh} refreshing={refreshing} />
        </div>
      ) : (
        <RunTerminal logs={logs} onClear={onClearLogs} />
      )}
    </section>
  );
}

function RunTerminal({ logs, onClear }: { logs: RuntimeLog[]; onClear: () => void }) {
  const { t, i18n } = useTranslation();
  const bodyRef = useRef<HTMLDivElement>(null);
  const language = i18n.language.startsWith('zh') ? 'zh' : 'en';
  const latestProgress = [...logs].reverse().find((log) => log.progress)?.progress;

  function stageLabel(progress: RuntimeProgress): string {
    const key = progress.stage.replace(/\s+/g, '_');
    return t(`terminal.stages.${key}`, { defaultValue: progress.stage });
  }

  function progressSummary(progress: RuntimeProgress): string {
    const percent = progress.percent === null ? '—' : `${Math.round(progress.percent)}%`;
    const count = progress.total === null
      ? formatNumber(progress.current, language)
      : `${formatNumber(progress.current, language)}/${formatNumber(progress.total, language)}`;
    const eta = progress.done
      ? t('terminal.progress.done')
      : t('terminal.progress.eta', { value: formatDuration(progress.eta_seconds) });
    return `${percent} · ${count} · ${t('terminal.progress.elapsed', { value: formatDuration(progress.elapsed_seconds) })} · ${eta}`;
  }

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logs.length]);

  return (
    <div className="run-terminal" aria-label={t('terminal.title')}>
      <div className="run-terminal-header">
        <div className="run-terminal-heading"><Terminal size={15} /><strong>{t('terminal.title')}</strong><span>{t('terminal.lines', { count: logs.length })}</span></div>
        <div className="run-terminal-actions">
          <span className="run-terminal-live"><span />{t('terminal.live')}</span>
          <button onClick={onClear} disabled={!logs.length}>{t('terminal.clear')}</button>
        </div>
      </div>

      {latestProgress && (
        <div className={`run-terminal-progress ${latestProgress.done ? 'done' : ''}`} aria-label={t('terminal.progress.label')}>
          <div className="run-terminal-progress-summary">
            <strong>{stageLabel(latestProgress)}</strong>
            <span>{progressSummary(latestProgress)}</span>
          </div>
          <div className="run-terminal-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={latestProgress.percent ?? undefined}>
            <span style={{ width: `${Math.max(0, Math.min(100, latestProgress.percent ?? 0))}%` }} />
          </div>
        </div>
      )}

      <div className="run-terminal-body" ref={bodyRef} role="log" aria-live="polite">
        {!logs.length && <div className="run-terminal-empty">{t('terminal.empty')}</div>}
        {logs.map((log) => (
          <div className={`run-terminal-line ${log.level}`} key={log.id}>
            <time>{formatTime(log.time, language)}</time>
            <span className="run-terminal-level">{t(`terminal.levels.${log.level}`, { defaultValue: log.level })}</span>
            <span className="run-terminal-message">{log.progress ? `${stageLabel(log.progress)} · ${progressSummary(log.progress)}` : log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssistantDrawer({
  overview,
  selectedImage,
  status,
  statusLoading,
  input,
  messages,
  pendingPlan,
  busy,
  onClose,
  onInputChange,
  onSubmit,
  onConfirm,
  onCancel,
  onQuickPrompt,
}: {
  overview: Overview | null;
  selectedImage: DatasetImage | null;
  status: VLMStatus | null;
  statusLoading: boolean;
  input: string;
  messages: AssistantMessage[];
  pendingPlan: AssistantResponse | null;
  busy: boolean;
  onClose: () => void;
  onInputChange: (value: string) => void;
  onSubmit: (event?: FormEvent) => void;
  onConfirm: () => void;
  onCancel: () => void;
  onQuickPrompt: (value: string) => void;
}) {
  const { t } = useTranslation();
  const statusLabel = statusLoading
    ? t('assistant.statusChecking')
    : status?.error
      ? t('assistant.statusUnavailable')
    : status?.configured
      ? t('assistant.statusReady')
      : t('assistant.statusNotConfigured');
  const statusClass = statusLoading ? 'checking' : status?.error ? 'not-ready' : status?.configured ? 'ready' : 'not-ready';

  return (
    <section className="assistant-panel" aria-label={t('assistant.title')}>
      <div className="assistant-header">
        <div className="assistant-heading">
          <div className="assistant-avatar"><Bot size={18} /></div>
          <div>
            <strong>{t('assistant.title')}</strong>
            <span>{t('assistant.subtitle')}</span>
          </div>
        </div>
        <button className="icon-button small" onClick={onClose} title={t('assistant.close')} aria-label={t('assistant.close')}><X size={16} /></button>
      </div>

      <div className="assistant-status-row">
        <span className={`assistant-status ${statusClass}`}><span className="assistant-status-dot" />{statusLabel}</span>
        {status?.model && <span className="assistant-model">{status.provider} · {status.model}</span>}
      </div>

      <div className="assistant-context">
        <div><span>{t('assistant.contextDataset')}</span><strong title={overview?.root || undefined}>{overview?.root || t('assistant.noDataset')}</strong></div>
        <div><span>{t('assistant.contextImage')}</span><strong title={selectedImage?.relative_path || undefined}>{selectedImage?.name || t('assistant.noImage')}</strong></div>
      </div>

      {!statusLoading && !status?.configured && <div className="assistant-hint">{t(status?.error ? 'assistant.apiUnavailableHint' : 'assistant.notConfiguredHint')}</div>}

      <div className="assistant-messages" aria-live="polite">
        {!messages.length && <div className="assistant-welcome"><div className="assistant-welcome-icon"><Bot size={20} /></div><p>{t('assistant.welcome')}</p></div>}
        {messages.map((message) => (
          <div className={`assistant-message ${message.role}`} key={message.id}>
            <div className="assistant-message-label">{message.role === 'user' ? t('assistant.you') : t('assistant.title')}</div>
            <div className="assistant-bubble">{message.content}</div>
            {message.response?.plan && <AssistantPlanSummary response={message.response} />}
          </div>
        ))}
        {busy && <div className="assistant-message assistant"><div className="assistant-message-label">{t('assistant.title')}</div><div className="assistant-bubble assistant-thinking"><span /><span /><span /></div></div>}
      </div>

      {pendingPlan && (
        <div className="assistant-plan-card">
          <div className="assistant-plan-title"><Check size={15} />{t('assistant.planTitle')}</div>
          <strong>{pendingPlan.plan.method}</strong>
          <span>{pendingPlan.plan.explanation || t('assistant.planReady')}</span>
          {pendingPlan.plan.requires_confirmation && <small>{t('assistant.requiresConfirmation')}</small>}
          <div className="assistant-plan-actions">
            <button className="primary-button" onClick={onConfirm} disabled={busy}><Check size={15} />{t('assistant.confirm')}</button>
            <button className="secondary-button" onClick={onCancel} disabled={busy}>{t('assistant.cancel')}</button>
          </div>
        </div>
      )}

      <div className="assistant-quick-prompts">
        <button onClick={() => onQuickPrompt(t('assistant.quickStats'))}>{t('assistant.quickStats')}</button>
        <button onClick={() => onQuickPrompt(t('assistant.quickCheck'))}>{t('assistant.quickCheck')}</button>
        <button onClick={() => onQuickPrompt(t('assistant.quickClasses'))}>{t('assistant.quickClasses')}</button>
      </div>

      <form className="assistant-composer" onSubmit={onSubmit}>
        <textarea
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          placeholder={t('assistant.inputPlaceholder')}
          disabled={busy || !overview}
          rows={2}
        />
        <button className="assistant-send" type="submit" disabled={busy || !input.trim() || !overview} title={t('assistant.send')} aria-label={t('assistant.send')}>
          {busy ? <RefreshCw className="spin" size={16} /> : <Send size={16} />}
        </button>
      </form>
      {!overview && <div className="assistant-composer-note">{t('assistant.loadDatasetFirst')}</div>}
    </section>
  );
}

function AssistantPlanSummary({ response }: { response: AssistantResponse }) {
  const { t } = useTranslation();
  return <div className="assistant-plan-summary"><span>{t('assistant.method')}: <code>{response.plan.method}</code></span>{response.executed && <span className="assistant-executed">{t('assistant.executed')}</span>}{response.result !== null && response.result !== undefined && <pre>{formatAssistantResult(response.result)}</pre>}</div>;
}

function formatAssistantResult(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2).slice(0, 2400);
  } catch {
    return String(value);
  }
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
