import React, { useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, Bone, Boxes, BrainCircuit, Check, ChevronDown, ChevronRight, Eye, EyeOff, Loader2, Play, RefreshCw, ScanLine, Search, X } from 'lucide-react';
import { analyzeSlices, getBoneFindings, getSlices, getVolumes, processTotalSeg, processVolumes, runBoneFindings } from './api.js';
import VolumeScene from './VolumeScene.jsx';

const ORGAN_GROUPS = [
  {
    id: 'lungs',
    label: 'Lungs',
    matches: (name) => name.startsWith('lung_'),
  },
  {
    id: 'heart',
    label: 'Heart',
    matches: (name) => name === 'heart' || name.includes('heart') || name.includes('atrial') || name.includes('ventricle'),
  },
  {
    id: 'arteries',
    label: 'Arteries',
    matches: (name) => name.includes('artery') || name.includes('aorta') || name.includes('trunk'),
  },
  {
    id: 'veins',
    label: 'Veins',
    matches: (name) => name.includes('vein') || name.includes('vena_cava'),
  },
  {
    id: 'bones',
    label: 'Bones',
    matches: (name) =>
      name.startsWith('rib_') ||
      name.startsWith('vertebrae_') ||
      name === 'sternum' ||
      name.startsWith('clavicula_') ||
      name.startsWith('scapula_') ||
      name.startsWith('humerus_') ||
      name === 'costal_cartilages',
  },
  {
    id: 'abdomen',
    label: 'Abdominal Organs',
    matches: (name) =>
      [
        'liver',
        'spleen',
        'stomach',
        'duodenum',
        'colon',
        'small_bowel',
        'kidney_left',
        'kidney_right',
        'pancreas',
        'gallbladder',
        'adrenal_gland_left',
        'adrenal_gland_right',
      ].includes(name),
  },
  {
    id: 'airway',
    label: 'Airway',
    matches: (name) => name === 'trachea' || name === 'esophagus',
  },
  {
    id: 'muscle',
    label: 'Muscle',
    matches: (name) => name.includes('muscle') || name.startsWith('autochthon') || name.startsWith('iliopsoas'),
  },
];

const CHEST_CORE_GROUP_IDS = ['lungs', 'heart', 'arteries', 'veins'];

function formatAnatomyLabel(label) {
  return label
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function anatomyGroupForLabel(label) {
  for (const group of ORGAN_GROUPS) {
    if (group.matches(label)) {
      return group.id;
    }
  }
  return 'other';
}

function SliceTile({ slice, selected, onToggle }) {
  return (
    <button type="button" className={`slice-tile ${selected ? 'selected' : ''}`} onClick={() => onToggle(slice.id)}>
      <img src={slice.thumbnail_url} alt={slice.id} loading="lazy" />
      <div className="slice-badge">{selected && <Check size={14} />}</div>
      <span>{slice.id.replace('ID_', '').replace('_CT', '')}</span>
      <small>{slice.contrast ? 'contrast' : 'noncontrast'} · age {slice.age ?? 'n/a'}</small>
    </button>
  );
}

function OrganGroup({ group, visibleSet, onToggleLabel, onToggleGroup, onShowOnly }) {
  const [collapsed, setCollapsed] = useState(false);
  const selectedCount = group.labels.filter((label) => visibleSet.has(label)).length;
  const allSelected = selectedCount === group.labels.length;
  const noneSelected = selectedCount === 0;

  return (
    <div className="organ-group-card">
      <div className="organ-group-head" onClick={() => setCollapsed((v) => !v)}>
        <div className="organ-group-left">
          <span className="organ-group-chevron">
            {collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
          </span>
          <strong>{group.label}</strong>
          <span className={`organ-group-count ${allSelected ? 'all' : noneSelected ? 'none' : ''}`}>
            {selectedCount}/{group.labels.length}
          </span>
        </div>
        <div className="organ-group-actions" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className={`organ-group-toggle ${allSelected ? 'active' : ''}`}
            onClick={() => onToggleGroup(group.labels)}
            title={allSelected ? `Hide ${group.label.toLowerCase()}` : `Show ${group.label.toLowerCase()}`}
          >
            {allSelected ? <EyeOff size={13} /> : <Eye size={13} />}
            <span>{allSelected ? 'Hide' : 'Show'}</span>
          </button>
          <button type="button" onClick={() => onShowOnly(group.labels)}>
            Only
          </button>
        </div>
      </div>
      {!collapsed && (
        <div className="organ-checklist">
          {group.labels.map((label) => {
            const visible = visibleSet.has(label);
            return (
              <label key={label} className={`organ-check-item ${visible ? 'active' : ''}`}>
                <input
                  type="checkbox"
                  checked={visible}
                  onChange={() => onToggleLabel(label)}
                />
                <span className="organ-check-box">
                  {visible && <Check size={11} strokeWidth={3} />}
                </span>
                <span className="organ-check-text">{formatAnatomyLabel(label)}</span>
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [slices, setSlices] = useState([]);
  const [volumes, setVolumes] = useState([]);
  const [activeVolumeId, setActiveVolumeId] = useState('');
  const [selectedSlices, setSelectedSlices] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [provider, setProvider] = useState('gemini');
  const [sceneKey, setSceneKey] = useState(0);
  const [visibleAnatomyLabels, setVisibleAnatomyLabels] = useState([]);
  const [organQuery, setOrganQuery] = useState('');
  const [organDropdownOpen, setOrganDropdownOpen] = useState(true);
  const [findings, setFindings] = useState([]);
  const [visibleFindingLabels, setVisibleFindingLabels] = useState([]);

  async function refresh() {
    setError('');
    const [sliceData, volumeData] = await Promise.all([getSlices(), getVolumes()]);
    setSlices(sliceData);
    setVolumes(volumeData);
    if (!activeVolumeId && volumeData.length) {
      setActiveVolumeId(volumeData[0].id);
    }
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  const activeVolume = useMemo(
    () => volumes.find((volume) => volume.id === activeVolumeId) || volumes[0],
    [volumes, activeVolumeId]
  );
  const anatomyLabels = useMemo(() => {
    if (activeVolume?.kind !== 'totalsegmentator') return [];
    const labels = (activeVolume.anatomy || [])
      .map((item) => item?.label)
      .filter((label) => typeof label === 'string' && label.length > 0);
    return Array.from(new Set(labels)).sort();
  }, [activeVolume]);
  const anatomyGroups = useMemo(() => {
    const byId = new Map();
    for (const group of ORGAN_GROUPS) {
      byId.set(group.id, { id: group.id, label: group.label, labels: [] });
    }
    byId.set('other', { id: 'other', label: 'Other', labels: [] });

    for (const label of anatomyLabels) {
      byId.get(anatomyGroupForLabel(label))?.labels.push(label);
    }
    return Array.from(byId.values()).filter((group) => group.labels.length > 0);
  }, [anatomyLabels]);
  const filteredAnatomyGroups = useMemo(() => {
    const needle = organQuery.trim().toLowerCase();
    if (!needle) return anatomyGroups;

    return anatomyGroups
      .map((group) => ({
        ...group,
        labels: group.labels.filter((label) => {
          const pretty = formatAnatomyLabel(label).toLowerCase();
          return label.toLowerCase().includes(needle) || pretty.includes(needle);
        }),
      }))
      .filter((group) => group.labels.length > 0);
  }, [anatomyGroups, organQuery]);
  const visibleAnatomySet = useMemo(() => new Set(visibleAnatomyLabels), [visibleAnatomyLabels]);
  const organFilteringEnabled = activeVolume?.kind === 'totalsegmentator' && anatomyLabels.length > 0;
  const selectedSliceData = useMemo(
    () => selectedSlices.map((id) => slices.find((slice) => slice.id === id)).filter(Boolean),
    [selectedSlices, slices]
  );
  const filteredOrganCount = useMemo(
    () => filteredAnatomyGroups.reduce((count, group) => count + group.labels.length, 0),
    [filteredAnatomyGroups]
  );

  useEffect(() => {
    if (!anatomyLabels.length) {
      setVisibleAnatomyLabels([]);
      return;
    }
    setVisibleAnatomyLabels(anatomyLabels);
  }, [anatomyLabels]);

  useEffect(() => {
    setOrganQuery('');
  }, [activeVolumeId]);

  useEffect(() => {
    if (!activeVolume?.id || activeVolume.kind !== 'totalsegmentator') {
      setFindings([]);
      setVisibleFindingLabels([]);
      return;
    }
    getBoneFindings(activeVolume.id)
      .then((data) => {
        const items = data?.findings || [];
        setFindings(items);
        setVisibleFindingLabels(items.map((f) => f.mesh_label));
      })
      .catch(() => {
        setFindings([]);
        setVisibleFindingLabels([]);
      });
  }, [activeVolume?.id, activeVolume?.kind]);

  const visibleFindingSet = useMemo(() => new Set(visibleFindingLabels), [visibleFindingLabels]);

  function toggleFindingLabel(meshLabel) {
    setVisibleFindingLabels((current) =>
      current.includes(meshLabel) ? current.filter((l) => l !== meshLabel) : [...current, meshLabel]
    );
  }

  async function handleRunBoneFindings() {
    if (!activeVolume?.id) return;
    setBusy('findings');
    setError('');
    try {
      const data = await runBoneFindings(activeVolume.id, { force: true });
      const items = data?.findings || [];
      setFindings(items);
      setVisibleFindingLabels(items.map((f) => f.mesh_label));
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  function toggleSlice(id) {
    setSelectedSlices((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-5)
    );
  }

  async function handleGenerate() {
    setBusy('process');
    setError('');
    try {
      const generated = await processVolumes(10);
      setVolumes(generated);
      setActiveVolumeId(generated[0]?.id || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function handleTotalSeg() {
    setBusy('totalseg');
    setError('');
    try {
      const generated = await processTotalSeg({ runSegmentation: true, fast: false });
      const volumeData = await getVolumes();
      setVolumes(volumeData);
      setActiveVolumeId(generated.id || volumeData[0]?.id || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function handleAnalyze(dryRun = false) {
    if (!selectedSlices.length) return;
    setBusy(dryRun ? 'dryrun' : 'analyze');
    setError('');
    try {
      setAnalysis(await analyzeSlices(selectedSlices, note, dryRun, provider));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  function handleAnalyzeSubmit(event) {
    event.preventDefault();
    if (busy === 'analyze' || busy === 'dryrun') return;
    void handleAnalyze(false);
  }

  function applyAnatomySelection(update) {
    setVisibleAnatomyLabels((current) => {
      const next = new Set(current);
      update(next);
      return anatomyLabels.filter((label) => next.has(label));
    });
  }

  function toggleAnatomyLabel(label) {
    applyAnatomySelection((selection) => {
      if (selection.has(label)) {
        selection.delete(label);
      } else {
        selection.add(label);
      }
    });
  }

  function toggleAnatomyGroup(groupLabels) {
    applyAnatomySelection((selection) => {
      const allVisible = groupLabels.every((label) => selection.has(label));
      if (allVisible) {
        groupLabels.forEach((label) => selection.delete(label));
      } else {
        groupLabels.forEach((label) => selection.add(label));
      }
    });
  }

  function focusChestCore() {
    const targetLabels = anatomyGroups
      .filter((group) => CHEST_CORE_GROUP_IDS.includes(group.id))
      .flatMap((group) => group.labels);
    setVisibleAnatomyLabels(targetLabels);
  }

  function showOnlyAnatomyLabels(labels) {
    const include = new Set(labels);
    setVisibleAnatomyLabels(anatomyLabels.filter((label) => include.has(label)));
  }

  function invertVisibleAnatomy() {
    setVisibleAnatomyLabels((current) => {
      const currentSet = new Set(current);
      return anatomyLabels.filter((label) => !currentSet.has(label));
    });
  }

  return (
    <main className="app-shell">
      <VolumeScene
        key={sceneKey}
        volume={activeVolume}
        onReset={() => setSceneKey((value) => value + 1)}
        organFilterEnabled={organFilteringEnabled}
        visibleLabels={[...visibleAnatomyLabels, ...visibleFindingLabels]}
      />
      <aside className="analysis-pane">
        <header className="topbar">
          <div>
            <h1>CT Spatial VLM Viewer</h1>
            <p>{slices.length} CT slices available</p>
          </div>
          <button type="button" title="Refresh data" onClick={() => refresh()} className="icon-button">
            <RefreshCw size={18} />
          </button>
        </header>

        <div className="action-row">
          <div className="action-buttons">
            <button type="button" onClick={handleGenerate} disabled={busy === 'process'}>
              {busy === 'process' ? <Loader2 className="spin" size={18} /> : <Boxes size={18} />}
              <span>Generate 10</span>
            </button>
            <button type="button" onClick={handleTotalSeg} disabled={busy === 'totalseg'}>
              {busy === 'totalseg' ? <Loader2 className="spin" size={18} /> : <ScanLine size={18} />}
              <span>TotalSeg HQ</span>
            </button>
          </div>
          <div className="volume-select-wrap">
            <label htmlFor="volume-select">Active 3D volume</label>
            <select id="volume-select" value={activeVolume?.id || ''} onChange={(event) => setActiveVolumeId(event.target.value)}>
              {volumes.length === 0 && <option>No volumes yet</option>}
              {volumes.map((volume) => (
                <option value={volume.id} key={volume.id}>
                  {volume.id}
                </option>
              ))}
            </select>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {organFilteringEnabled && (
          <section className="organ-panel">
            <div className="organ-panel-header">
              <div className="organ-panel-header-left">
                <Boxes size={16} />
                <h2>Organ Visibility</h2>
              </div>
              <span className="organ-count-badge">
                {visibleAnatomyLabels.length}/{anatomyLabels.length} shown
              </span>
            </div>

            <div className="organ-panel-body">
              <button
                type="button"
                className={`organ-dropdown-trigger ${organDropdownOpen ? 'open' : ''}`}
                onClick={() => setOrganDropdownOpen((value) => !value)}
                aria-expanded={organDropdownOpen}
              >
                <span className="organ-dropdown-trigger-left">
                  {organDropdownOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  Detected organ list
                </span>
                <small>{visibleAnatomyLabels.length}/{anatomyLabels.length} visible</small>
              </button>

              {organDropdownOpen && (
                <div className="organ-dropdown-content">
                  <div className="organ-detected-note">
                    <span>{anatomyLabels.length} organs detected in the active 3D volume</span>
                    <span>{filteredOrganCount} shown in this list</span>
                  </div>
                  <div className="organ-quick-actions">
                    <button type="button" onClick={() => setVisibleAnatomyLabels(anatomyLabels)} disabled={visibleAnatomyLabels.length === anatomyLabels.length}>
                      Show all
                    </button>
                    <button type="button" onClick={() => setVisibleAnatomyLabels([])} disabled={visibleAnatomyLabels.length === 0}>
                      Hide all
                    </button>
                    <button type="button" onClick={invertVisibleAnatomy} disabled={anatomyLabels.length === 0}>
                      Invert
                    </button>
                    <button
                      type="button"
                      onClick={focusChestCore}
                      disabled={!anatomyGroups.some((group) => CHEST_CORE_GROUP_IDS.includes(group.id))}
                    >
                      Chest core
                    </button>
                  </div>

                  <div className="organ-search-row">
                    <Search size={14} />
                    <input
                      type="text"
                      value={organQuery}
                      onChange={(event) => setOrganQuery(event.target.value)}
                      placeholder="Search organs, vessels, bones..."
                    />
                    {organQuery && (
                      <button type="button" className="organ-search-clear" onClick={() => setOrganQuery('')}>
                        <X size={13} />
                      </button>
                    )}
                  </div>

                  <div className="organ-scroll-list">
                    {filteredAnatomyGroups.length === 0 ? (
                      <div className="organ-empty">No matching anatomy found.</div>
                    ) : (
                      filteredAnatomyGroups.map((group) => (
                        <OrganGroup
                          key={group.id}
                          group={group}
                          visibleSet={visibleAnatomySet}
                          onToggleLabel={toggleAnatomyLabel}
                          onToggleGroup={toggleAnatomyGroup}
                          onShowOnly={showOnlyAnatomyLabels}
                        />
                      ))
                    )}
                  </div>
                </div>
              )}
              </div>
          </section>
        )}

        {organFilteringEnabled && (
          <section className="findings-panel">
            <div className="findings-panel-header">
              <div className="findings-panel-header-left">
                <Bone size={16} />
                <h2>Bone Findings</h2>
              </div>
              <div className="findings-panel-header-right">
                {findings.length > 0 && (
                  <span className="findings-count-badge">
                    {findings.length} candidate{findings.length !== 1 ? 's' : ''}
                  </span>
                )}
                <button
                  type="button"
                  className="findings-run-btn"
                  onClick={handleRunBoneFindings}
                  disabled={busy === 'findings'}
                  title="Run bone fracture detection"
                >
                  {busy === 'findings' ? <Loader2 className="spin" size={14} /> : <Play size={14} />}
                  <span>Detect</span>
                </button>
              </div>
            </div>

            {findings.length === 0 ? (
              <div className="findings-empty">
                <AlertTriangle size={16} />
                <span>No bone findings detected yet. Press Detect to run analysis.</span>
              </div>
            ) : (
              <div className="findings-body">
                <div className="findings-disclaimer">
                  <AlertTriangle size={13} />
                  <span>Algorithmic candidates only — radiologist review required</span>
                </div>
                <div className="findings-list">
                  {findings.map((finding) => {
                    const visible = visibleFindingSet.has(finding.mesh_label);
                    const isHighConf = finding.confidence >= 0.70;
                    return (
                      <label
                        key={finding.id}
                        className={`finding-card ${visible ? 'active' : ''} ${isHighConf ? 'high-conf' : 'med-conf'}`}
                      >
                        <input
                          type="checkbox"
                          checked={visible}
                          onChange={() => toggleFindingLabel(finding.mesh_label)}
                        />
                        <span className="finding-check-box">
                          {visible && <Check size={11} strokeWidth={3} />}
                        </span>
                        <div className="finding-info">
                          <span className="finding-label">{finding.label}</span>
                          <span className="finding-bone">{formatAnatomyLabel(finding.bone_label)}</span>
                        </div>
                        <span className={`finding-confidence ${isHighConf ? 'high' : 'med'}`}>
                          {Math.round(finding.confidence * 100)}%
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
          </section>
        )}

        <section className="vlm-panel">
          <div className="section-title">
            <BrainCircuit size={18} />
            <h2>Medical VLM Analysis</h2>
            <span>{selectedSlices.length}/5 selected</span>
          </div>
          <form className="vlm-form" onSubmit={handleAnalyzeSubmit}>
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask the selected VLM about the chosen CT images"
              rows={4}
            />
            <div className="analysis-controls">
              <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                <option value="gemini">Gemini Lite</option>
                <option value="groq">Groq</option>
              </select>
              <button type="submit" disabled={!selectedSlices.length || busy === 'analyze'}>
                {busy === 'analyze' ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                <span>Analyze</span>
              </button>
              <button type="button" onClick={() => handleAnalyze(true)} disabled={!selectedSlices.length || busy === 'dryrun'}>
                {busy === 'dryrun' ? <Loader2 className="spin" size={18} /> : <BrainCircuit size={18} />}
                <span>Dry run</span>
              </button>
            </div>
          </form>
          <div className="selected-strip">
            {selectedSliceData.length === 0 ? (
              <div className="selected-empty">Select CT slices below</div>
            ) : (
              selectedSliceData.map((slice) => (
                <button type="button" className="selected-chip" key={slice.id} onClick={() => toggleSlice(slice.id)}>
                  <img src={slice.thumbnail_url} alt={slice.id} />
                  <span>{slice.id.replace('ID_', '').replace('_CT', '')}</span>
                  <X size={14} />
                </button>
              ))
            )}
          </div>
          {analysis && (
            <div className="analysis-output">
              <strong>{analysis.provider || provider} · {analysis.model}</strong>
              <p>{analysis.analysis}</p>
            </div>
          )}
        </section>

        <section className="slice-section">
          <div className="section-title">
            <Activity size={18} />
            <h2>2D CT Slices</h2>
          </div>
          <div className="slice-grid">
            {slices.map((slice) => (
              <SliceTile
                key={slice.id}
                slice={slice}
                selected={selectedSlices.includes(slice.id)}
                onToggle={toggleSlice}
              />
            ))}
          </div>
        </section>
      </aside>
    </main>
  );
}
