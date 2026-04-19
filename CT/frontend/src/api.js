const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export function getSlices() {
  return request('/api/slices?limit=100');
}

export function getVolumes() {
  return request('/api/volumes');
}

export function processVolumes(limit = 10) {
  return request(`/api/process?limit=${limit}`, { method: 'POST' });
}

export function processTotalSeg({ runSegmentation = true, forceSegmentation = false, device = 'mps', fast = true } = {}) {
  const params = new URLSearchParams({
    run_segmentation: String(runSegmentation),
    force_segmentation: String(forceSegmentation),
    device,
    fast: String(fast)
  });
  return request(`/api/process-totalseg?${params.toString()}`, { method: 'POST' });
}

export function analyzeSlices(sliceIds, userNote = '', dryRun = false, provider = 'groq') {
  return request('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slice_ids: sliceIds, user_note: userNote, dry_run: dryRun, provider })
  });
}

export function getBoneFindings(sourceId) {
  return request(`/api/findings/bones/${encodeURIComponent(sourceId)}`);
}

export function runBoneFindings(sourceId, { force = false, minConfidence = 0.55 } = {}) {
  const params = new URLSearchParams({
    source_id: sourceId,
    force: String(force),
    min_confidence: String(minConfidence),
  });
  return request(`/api/findings/bones/run?${params.toString()}`, { method: 'POST' });
}
