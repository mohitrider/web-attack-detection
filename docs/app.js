(() => {
  const $ = (id) => document.getElementById(id);
  const fmtInt = (n) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toLocaleString('en-US');
  const fmtPct = (n, digits = 1) => n == null || !Number.isFinite(Number(n)) ? '—' : `${(Number(n) * 100).toFixed(digits)}%`;
  const fmt = (n, digits = 3) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toFixed(digits);
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  let D = null;

  function metricLabel(metric) {
    return ({F1:'F1-score', Precision:'Precision', Recall:'Recall', PR_AUC_AveragePrecision:'Average precision (PR-AUC)', ROC_AUC:'ROC-AUC', FP:'False positives', FN:'False negatives'})[metric] || metric;
  }
  function directionLabel(direction) {
    return direction === 'Thursday_to_Friday' ? 'Thursday → Friday' : 'Friday → Thursday';
  }
  function modelShort(name) { return name === 'XGBoost' ? 'XGBoost' : name; }
  function recordsForDirection(rows, direction) { return rows.filter(r => r.Direction === direction); }
  function table(headers, rows, className = '') {
    if (!rows.length) return '<div class="empty-state">No records available for this view.</div>';
    return `<div class="table-wrap"><table class="data-table ${className}"><thead><tr>${headers.map(h => `<th>${escapeHtml(h.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map(h => {
      const value = row[h.key];
      let output = h.render ? h.render(value, row) : escapeHtml(value);
      return `<td class="${h.className ? h.className(row) : ''}">${output}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  function kpi(label, value, description, icon) {
    return `<div class="kpi-card"><div class="kpi-top"><span>${escapeHtml(label)}</span><span class="kpi-icon">${icon}</span></div><div class="kpi-value">${value}</div><div class="kpi-description">${escapeHtml(description)}</div></div>`;
  }
  function miniBars(container, rows, valueKey, display, maximum = null) {
    if (!rows.length) { container.innerHTML = '<div class="empty-state">No data available.</div>'; return; }
    const values = rows.map(r => Number(r[valueKey])).filter(Number.isFinite);
    const maxVal = maximum ?? Math.max(...values, 0);
    container.innerHTML = rows.map(row => {
      const value = Number(row[valueKey]);
      const width = Number.isFinite(value) && maxVal > 0 ? Math.max(1.5, value / maxVal * 100) : 0;
      return `<div class="bar-row"><div class="bar-meta"><strong>${escapeHtml(row.Model || row.Variant || row.name)}</strong><span>${escapeHtml(display(value, row))}</span></div><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, width)}%"></div></div>${row.sub ? `<div class="bar-sub">${escapeHtml(row.sub)}</div>` : ''}</div>`;
    }).join('');
  }
  function metricRows() { return D.modelMetrics || []; }

  function renderOverview() {
    const a = D.audit;
    $('overview-kpis').innerHTML = [
      kpi('Network flows', fmtInt(a.totalFlows), 'Two original traffic-day files', '▤'),
      kpi('Web attacks', fmtInt(a.attackFlows), `${fmtPct(a.attackFlows / a.totalFlows, 3)} of all flows`, '⚑'),
      kpi('Flow features', fmtInt(a.featuresInRawFiles), 'Timestamp and label excluded from model inputs', '⌘'),
      kpi('Models compared', '03', 'Logistic Regression · Linear SVM · XGBoost', '↗'),
    ].join('');
    const dirs = ['Thursday_to_Friday','Friday_to_Thursday'];
    $('overview-direction-cards').innerHTML = dirs.map(dir => {
      const rows = recordsForDirection(metricRows(), dir);
      const best = [...rows].sort((x,y) => Number(y.F1)-Number(x.F1))[0];
      return `<div class="direction-card"><div class="direction-title"><b>↗</b>${directionLabel(dir)}</div><div class="direction-best">${best ? fmt(best.F1,3) : '—'}</div><div class="direction-caption">Best F1-score on the opposite-day test file</div><span class="direction-model">${best ? escapeHtml(best.Model) : 'No result'}</span></div>`;
    }).join('');
    const xgb = metricRows().filter(r => r.Model === 'XGBoost');
    const first = xgb.find(r => r.Direction === 'Thursday_to_Friday');
    const second = xgb.find(r => r.Direction === 'Friday_to_Thursday');
    if (first && second) $('overview-insight').textContent = `XGBoost F1 is ${fmt(first.F1)} for Thursday → Friday and ${fmt(second.F1)} for Friday → Thursday. This directional difference makes generalisation a central result, not just a model leaderboard.`;
    const distRows = [
      {name:'Benign', count:a.benignFlows, cls:'benign'},
      {name:'Brute Force-Web', count:611, cls:'attack'},
      {name:'Brute Force-XSS', count:230, cls:'attack'},
      {name:'SQL Injection', count:87, cls:'attack'},
    ];
    $('class-distribution').innerHTML = distRows.map(row => `<div class="distribution-row ${row.cls}"><div class="distribution-head"><strong>${escapeHtml(row.name)}</strong><span>${fmtInt(row.count)}</span></div><div class="track"><span style="width:${Math.max(0.7,row.count/a.totalFlows*100)}%"></span></div></div>`).join('');
  }

  function renderModels() {
    const direction = $('model-direction').value;
    const metric = $('model-metric').value;
    const rows = recordsForDirection(metricRows(), direction);
    $('model-chart-title').textContent = `${metricLabel(metric)} by model`;
    miniBars($('model-bars'), rows, metric, value => metric === 'FP' || metric === 'FN' ? fmtInt(value) : fmt(value, 3));
    const headers = [
      {key:'Model',label:'Model',render:v=>`<span class="model-dot"></span>${escapeHtml(v)}`},
      {key:'Precision',label:'Precision',render:v=>fmtPct(v,2)},
      {key:'Recall',label:'Recall',render:v=>fmtPct(v,2)},
      {key:'F1',label:'F1',render:v=>fmt(v,3),className:r=>r.Model==='XGBoost'?'strong':''},
      {key:'PR_AUC_AveragePrecision',label:'PR-AUC',render:v=>fmt(v,3)},
      {key:'FP',label:'FP',render:fmtInt},
      {key:'FN',label:'FN',render:fmtInt},
    ];
    $('model-table-wrap').innerHTML = table(headers, rows);
    const attackRows = rows.map(row => ({Model:row.Model, web:row['Recall_Brute Force -Web'], xss:row['Recall_Brute Force -XSS'], sql:row['Recall_SQL Injection']}));
    $('attack-recall-table').innerHTML = table([
      {key:'Model',label:'Model'},
      {key:'web',label:'Brute Force-Web',render:fmtPct},
      {key:'xss',label:'Brute Force-XSS',render:fmtPct},
      {key:'sql',label:'SQL Injection',render:fmtPct},
    ], attackRows);
  }

  function renderAblation() {
    const direction = $('ablation-direction').value;
    const rows = (D.featureAblation || []).filter(r => r.Direction === direction);
    const baseline = rows.find(r => r.Variant === 'All usable features');
    const best = [...rows].sort((a,b)=>Number(b.Test_F1)-Number(a.Test_F1))[0];
    $('ablation-summary').innerHTML = [
      kpi('Reference F1', fmt(baseline?.Test_F1), 'All usable training-selected features', '◎'),
      kpi('Best variant F1', fmt(best?.Test_F1), best?.Variant || '—', '✳'),
      kpi('Features used', fmtInt(best?.Feature_Count), 'Feature count of the best F1 variant', '⌘'),
    ].join('');
    $('ablation-table').innerHTML = table([
      {key:'Variant',label:'Feature set'},
      {key:'Feature_Count',label:'Features',render:fmtInt},
      {key:'Test_Precision',label:'Precision',render:fmtPct},
      {key:'Test_Recall',label:'Recall',render:fmtPct},
      {key:'Test_F1',label:'F1',render:fmt, className:r=>r.Variant===best?.Variant?'strong':''},
      {key:'Test_Average_Precision',label:'PR-AUC',render:v=>fmt(v,3)},
      {key:'FP',label:'FP',render:fmtInt},
      {key:'FN',label:'FN',render:fmtInt},
    ], rows);
  }

  function renderImbalance() {
    const direction = $('imbalance-direction').value;
    const model = $('imbalance-model').value;
    const rows = (D.imbalanceComparison || []).filter(r => r.Direction === direction && r.Model === model);
    const headers = [
      {key:'Strategy',label:'Training strategy'},
      {key:'Train_Rows',label:'Rows',render:fmtInt},
      {key:'Train_Benign',label:'Benign',render:fmtInt},
      {key:'Train_Attacks',label:'Attacks',render:fmtInt},
      {key:'Precision',label:'Precision',render:fmtPct},
      {key:'Recall',label:'Recall',render:fmtPct},
      {key:'F1',label:'F1',render:fmt,className:r=>r.Strategy==='Current undersampling + class weights'?'strong':''},
      {key:'Average_Precision',label:'PR-AUC',render:v=>fmt(v,3)},
      {key:'FP',label:'FP',render:fmtInt},
      {key:'FN',label:'FN',render:fmtInt},
    ];
    $('imbalance-table').innerHTML = table(headers, rows);
  }

  function renderPatterns() {
    const direction = $('pattern-direction').value;
    const model = $('pattern-model').value;
    const rows = (D.seenUnseen || []).filter(r => r.Direction === direction && String(r.Model).replaceAll(' ','_') === model);
    const seen = rows.find(r => String(r.FeaturePatternGroup).toLowerCase().startsWith('seen'));
    const unseen = rows.find(r => String(r.FeaturePatternGroup).toLowerCase().startsWith('unseen'));
    $('pattern-cards').innerHTML = [
      kpi('Seen flows', fmtInt(seen?.Flows), `${fmtInt(seen?.Attacks)} attack flows`, '◉'),
      kpi('Unseen flows', fmtInt(unseen?.Flows), `${fmtInt(unseen?.Attacks)} attack flows`, '◎'),
      kpi('Unseen-pattern F1', fmt(unseen?.F1), model.replaceAll('_',' '), '↗'),
    ].join('');
    $('pattern-table').innerHTML = table([
      {key:'FeaturePatternGroup',label:'Pattern group'},
      {key:'Flows',label:'Flows',render:fmtInt},
      {key:'Attacks',label:'Attacks',render:fmtInt},
      {key:'Precision',label:'Precision',render:fmtPct},
      {key:'Recall',label:'Recall',render:fmtPct},
      {key:'F1',label:'F1',render:fmt},
      {key:'AveragePrecision',label:'PR-AUC',render:v=>fmt(v,3)},
      {key:'FP',label:'FP',render:fmtInt},
      {key:'FN',label:'FN',render:fmtInt},
    ], rows);
  }

  function renderData() {
    const a = D.audit;
    $('audit-kpis').innerHTML = [
      kpi('Missing feature values', fmtInt(a.missingFeatureValues), 'Across the two original files', '∅'),
      kpi('Infinite feature values', fmtInt(a.infiniteFeatureValues), 'In feature columns', '∞'),
      kpi('Repeated feature groups', fmtInt(a.repeatedFeaturePatternGroups), 'Timestamp and label excluded', '⤢'),
      kpi('Conflicting-label groups', fmtInt(a.conflictingLabelSignatures), 'Includes benign/attack conflicts', '⚑'),
    ].join('');
    $('day-table').innerHTML = table([
      {key:'day',label:'Traffic day'},
      {key:'total',label:'Total flows',render:fmtInt},
      {key:'Benign',label:'Benign',render:fmtInt},
      {key:'Brute Force -Web',label:'BF-Web',render:fmtInt},
      {key:'Brute Force -XSS',label:'XSS',render:fmtInt},
      {key:'SQL Injection',label:'SQLi',render:fmtInt},
      {key:'attacks',label:'Total attacks',render:fmtInt},
    ], a.days);
    const q = [
      ['Exact duplicate raw rows (within files)',a.exactDuplicateRowsWithinFiles],
      ['Repeated feature-pattern groups',a.repeatedFeaturePatternGroups],
      ['Shared feature signatures across days',a.crossDaySharedFeatureSignatures],
      ['Constant features detected',a.constantFeaturesDetected],
      ['Conflicting-label signatures',a.conflictingLabelSignatures],
      ['Benign/attack conflict signatures',a.benignAttackConflictSignatures],
    ];
    $('quality-list').innerHTML = q.map(([label,value])=>`<div class="quality-item"><span>${escapeHtml(label)}</span><strong>${fmtInt(value)}</strong></div>`).join('');
  }

  function bindControls() {
    ['model-direction','model-metric'].forEach(id=>$(id).addEventListener('change',renderModels));
    $('ablation-direction').addEventListener('change',renderAblation);
    ['imbalance-direction','imbalance-model'].forEach(id=>$(id).addEventListener('change',renderImbalance));
    ['pattern-direction','pattern-model'].forEach(id=>$(id).addEventListener('change',renderPatterns));
    document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>switchView(button.dataset.view)));
  }
  function switchView(view) {
    document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
    document.querySelectorAll('.view').forEach(s=>s.classList.toggle('active',s.id===`view-${view}`));
    window.scrollTo({top:0,behavior:'smooth'});
  }

  async function init() {
    bindControls();
    try {
      const response = await fetch('data/dashboard_data.json', {cache:'no-store'});
      if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
      D = await response.json();
      renderOverview(); renderModels(); renderAblation(); renderImbalance(); renderPatterns(); renderData();
      $('load-status').textContent = '';
      $('footer-record-count').textContent = `${D.modelMetrics.length} primary model records`;
    } catch (error) {
      $('load-error').classList.remove('hidden');
      $('load-status').textContent = 'Could not load exported experiment data.';
      console.error(error);
    }
  }
  document.addEventListener('DOMContentLoaded',init);
})();
