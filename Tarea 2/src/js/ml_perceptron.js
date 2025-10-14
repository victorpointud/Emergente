// src/js/ml_perceptron.js
// UI helpers
let mlpConfigText = null;
const $ = (id) => document.getElementById(id);

// -------------------------
// Utils (CSV / files / API)
// -------------------------
function parseCSVLine(line){
  return line.split(',').map(s => s.trim()).filter(s => s.length > 0);
}

function readLocalFile(file){
  return new Promise((resolve, reject) => {
    if(!file) return reject(new Error("No file selected"));
    const r = new FileReader();
    r.onload  = () => resolve(r.result);
    r.onerror = () => reject(new Error("Error reading file"));
    r.readAsText(file, 'utf-8');
  });
}

async function api(path, payload){
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const txt = await res.text();
  let data = {};
  try { data = JSON.parse(txt); } catch { data = { error: "Invalid response" }; }
  if(!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function toRowsFromCSVText(t){
  const lines = t.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  const rows = [];
  for(const line of lines){
    const parts = parseCSVLine(line).map(v => Number(v));
    if(parts.length === 0) continue;
    if(parts.some(v => Number.isNaN(v))) continue;
    rows.push(parts);
  }
  return rows;
}

function parseHiddenNeuronsSafe(text, nHidden){
  const raw = (text || '').trim();
  let arr = [];
  if (!raw) {
    arr = Array(nHidden).fill(4);
  } else if (raw.includes(',')) {
    arr = raw.split(',').map(s => Math.max(1, parseInt(s.trim(), 10) || 4));
  } else {
    arr = Array(nHidden).fill(Math.max(1, parseInt(raw, 10) || 4));
  }
  if (arr.length < nHidden) {
    arr = [...arr, ...Array(nHidden - arr.length).fill(arr[arr.length - 1] || 4)];
  } else if (arr.length > nHidden) {
    arr = arr.slice(0, nHidden);
  }
  return arr;
}

function splitXYFromCombined(rows){
  const X = [], Y = [];
  for (const r of rows) {
    if (!Array.isArray(r) || r.length < 2) {
      throw new Error("Each row needs ≥2 columns (n-1 features + 1 label).");
    }
    X.push(r.slice(0, -1));
    Y.push([r[r.length - 1]]);
  }
  return { X, Y };
}

// -------------------------
// Create / Load MLP
// -------------------------
async function onCreate(){
  const n_in = Math.max(1, parseInt($('nIn').value, 10) || 1);
  const n_out = Math.max(1, parseInt($('nOut').value, 10) || 1);
  const n_hidden = Math.max(0, parseInt($('nHidden').value, 10) || 0);
  const hiddenNeurons = parseHiddenNeuronsSafe($('hiddenNeurons').value, n_hidden);
  const actHidden = $('actHidden').value;
  const actOut = $('actOut').value;

  const resp = await api('/api/mlp/create', { n_in, n_out, n_hidden, hidden_neurons: hiddenNeurons });
  $('createPreview').textContent = `Created. Sizes: [${resp.sizes.join(', ')}], Hidden: ${actHidden}, Output: ${actOut}`;
  // Al crear, aún no hay gráficos: desactiva TEST/History
  toggleGraphButtons(false, false);
}

function onPickMLPFile(){
  const f = $('fileMLP').files[0] || null;
  mlpConfigText = null;
  $('mlpPreview').textContent = f ? 'File selected. Press “Apply configuration”.' : '';
  if(f){
    readLocalFile(f)
      .then(t => mlpConfigText = t)
      .catch(e => $('mlpPreview').textContent = `Error: ${e.message}`);
  }
}

async function onApplyMLP(){
  if(!mlpConfigText){ $('mlpPreview').textContent = 'Select a configuration file.'; return; }
  try{
    const cfg = JSON.parse(mlpConfigText);
    await api('/api/mlp/apply_config', cfg);
    $('mlpPreview').textContent = `Applied. Sizes: [${cfg.sizes.join(', ')}]`;
    // Config aplicada: aún no hay history
    toggleGraphButtons(false, false);
    clearGraphs();
  }catch(err){
    $('mlpPreview').textContent = `Error: ${err.message}`;
  }
}

function actPair(){
  return { act_hidden: $('actHidden').value, act_out: $('actOut').value };
}

// -------------------------
// Train / Continue
// -------------------------
async function onTrain(){
  try{
    const { act_hidden } = actPair();
    const { act_out } = actPair();

    const epochs = Math.max(1, parseInt($('epochs').value, 10) || 1);
    const lr = parseFloat($('lr').value) || 0.05;

    const XtrFile = $('fileXtr').files[0];
    const YtrFile = $('fileYtr').files[0];
    const XteFile = $('fileXte').files[0];
    const YteFile = $('fileYte').files[0];

    let Xtr = [], Ytr = [], Xte = [], Yte = [];

    if (XtrFile && YtrFile) {
      Xtr = toRowsFromCSVText(await readLocalFile(XtrFile));
      Ytr = toRowsFromCSVText(await readLocalFile(YtrFile));
    } else if (XtrFile) {
      const comb = toRowsFromCSVText(await readLocalFile(XtrFile));
      const s = splitXYFromCombined(comb);
      Xtr = s.X; Ytr = s.Y;
    } else {
      $('trainLog').textContent = 'Provide Train X (+ optional Train Y) OR a single combined Train CSV.';
      return;
    }

    if (XteFile && YteFile) {
      Xte = toRowsFromCSVText(await readLocalFile(XteFile));
      Yte = toRowsFromCSVText(await readLocalFile(YteFile));
    } else if (XteFile) {
      const combT = toRowsFromCSVText(await readLocalFile(XteFile));
      const sT = splitXYFromCombined(combT);
      Xte = sT.X; Yte = sT.Y;
    }

    const resp = await api('/api/mlp/train', {
      X_train: Xtr, Y_train: Ytr,
      X_test:  Xte, Y_test:  Yte,
      epochs, lr, act_hidden, act_out
    });

    // Tabla de history
    const hist = resp.history || [];
    let html = '<table><thead><tr><th>Epoch</th><th>Train Acc</th><th>Test Acc</th></tr></thead><tbody>';
    for(const h of hist){
      const ta = (h.train_acc != null) ? h.train_acc.toFixed(4) : '—';
      const te = (h.test_acc  != null) ? h.test_acc.toFixed(4)  : '—';
      html += `<tr><td>${h.epoch}</td><td>${ta}</td><td>${te}</td></tr>`;
    }
    html += '</tbody></table>';
    $('trainLog').innerHTML = html;

    // Habilitar botones de gráficos:
    const canTrainPlots = hist.length > 0;
    const canTestPlots  = (Xte && Xte.length > 0) || (Yte && Yte.length > 0);
    toggleGraphButtons(canTrainPlots, canTestPlots);

  }catch(e){
    $('trainLog').textContent = `Error: ${e.message}`;
  }
}

async function onContinue(){
  return onTrain();
}

// -------------------------
// Evaluate
// -------------------------
function onEvalOneMLP(){
  const xTxt = $('inputVec').value.trim();
  if(!xTxt){ $('oneOutMLP').textContent = 'Enter a vector.'; return; }
  const x = parseCSVLine(xTxt).map(Number);
  api('/api/mlp/eval_one', { x })
    .then(({ y }) => $('oneOutMLP').textContent = `y = [${y.map(v => Number(v).toFixed(6)).join(', ')}]`)
    .catch(e => $('oneOutMLP').textContent = `Error: ${e.message}`);
}

async function onEvalFileMLP(){
  const f = $('fileInputsMLP').files[0];
  if(!f){ $('manyOutMLP').textContent = 'Select a CSV file.'; return; }
  try{
    const rows = toRowsFromCSVText(await readLocalFile(f));
    const resp = await api('/api/mlp/eval_file', { rows });
    const items = resp.results || [];
    let html = '<table><thead><tr><th>#</th><th>x</th><th>y</th><th>Status</th></tr></thead><tbody>';
    for(const it of items){
      const y = (typeof it.y === 'number')
        ? it.y
        : (Array.isArray(it.y) ? `[${it.y.map(v => Number(v).toFixed(6)).join(', ')}]` : '—');
      html += `<tr><td>${it.idx}</td><td>[${(it.x || []).join(', ')}]</td><td>${y}</td><td>${it.err ? ('Error: ' + it.err) : 'OK'}</td></tr>`;
    }
    html += '</tbody></table>';
    $('manyOutMLP').innerHTML = html;

    // Al evaluar archivo (test), ya puedes ver gráficos de TEST
    toggleGraphButtons(true, true);
  }catch(e){
    $('manyOutMLP').textContent = `Error: ${e.message}`;
  }
}

// -------------------------
// Save config
// -------------------------
async function onSave(){
  try{
    const obj = await api('/api/mlp/save', {});
    const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mlp_saved.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }catch(e){
    alert('Error: ' + e.message);
  }
}

// -------------------------
// Graphs (robustos)
// -------------------------
function setBoxState(boxId, imgId, msgId, state, text = ""){
  const box = $(boxId), img = $(imgId), msg = $(msgId);
  if (!box || !img || !msg) return;

  box.classList.remove("graph-loading");
  msg.style.display = "none";
  img.style.display = "none";

  if (state === "loading"){
    msg.textContent = "Loading…";
    msg.style.display = "flex";
    box.classList.add("graph-loading");
  } else if (state === "error"){
    msg.textContent = text || "Not available yet";
    msg.style.display = "flex";
  } else if (state === "ok"){
    img.style.display = "block";
  }
}

function clearGraphs(){
  setBoxState("boxData", "imgData", "msgData", "error", "No plot yet.");
  setBoxState("boxPred", "imgPred", "msgPred", "error", "No plot yet.");
  setBoxState("boxConf", "imgConf", "msgConf", "error", "Open TEST graphs to see confusion matrix");
  setBoxState("boxHist", "imgHistory2", "msgHist", "error", "No history yet.");
  // limpia objectURL si quedó alguno
  for (const id of ["imgData","imgPred","imgConf","imgHistory2"]){
    const img = $(id);
    if (img && img.dataset && img.dataset.url){
      URL.revokeObjectURL(img.dataset.url);
      delete img.dataset.url;
    }
    if (img) img.removeAttribute("src");
  }
}

async function loadPNGInto(url, boxId, imgId, msgId){
  try{
    setBoxState(boxId, imgId, msgId, "loading");
    const res = await fetch(url + (url.includes("?") ? "&" : "?") + "__t=" + Date.now(), { cache: "no-store" });
    if(!res.ok) {
      setBoxState(boxId, imgId, msgId, "error",
        res.status === 404 ? "Plot not ready. Train / provide TEST first." : "Error " + res.status
      );
      return false;
    }
    const blob = await res.blob();
    const img = $(imgId);

    // limpia objectURL previo si existía
    if (img.dataset.url) { URL.revokeObjectURL(img.dataset.url); delete img.dataset.url; }

    const objURL = URL.createObjectURL(blob);
    img.onload = () => {
      setBoxState(boxId, imgId, msgId, "ok");
      // revoca más tarde para no romper Safari
      setTimeout(() => {
        URL.revokeObjectURL(objURL);
        if (img.dataset.url === objURL) delete img.dataset.url;
      }, 3000);
    };
    img.onerror = () => {
      setBoxState(boxId, imgId, msgId, "error", "Failed to render image");
    };
    img.src = objURL;
    img.dataset.url = objURL;
    return true;
  }catch(e){
    setBoxState(boxId, imgId, msgId, "error", e.message);
    return false;
  }
}

async function onPlotTrain(){
  await loadPNGInto("/api/mlp/plot_data_train.png", "boxData", "imgData", "msgData");
  await loadPNGInto("/api/mlp/plot_pred_train.png", "boxPred", "imgPred", "msgPred");
  await loadPNGInto("/api/mlp/plot_history.png",     "boxHist", "imgHistory2", "msgHist");
  setBoxState("boxConf", "imgConf", "msgConf", "error", "Open TEST graphs to see confusion matrix");
}

async function onPlotTest(){
  await loadPNGInto("/api/mlp/plot_data_test.png",   "boxData", "imgData", "msgData");
  await loadPNGInto("/api/mlp/plot_pred_test.png",   "boxPred", "imgPred", "msgPred");
  await loadPNGInto("/api/mlp/plot_confusion.png",   "boxConf", "imgConf", "msgConf");
  await loadPNGInto("/api/mlp/plot_history.png",     "boxHist", "imgHistory2", "msgHist");
}

async function onPlotHistory(){
  setBoxState("boxData", "imgData", "msgData", "error", "History view doesn’t show dataset.");
  setBoxState("boxPred", "imgPred", "msgPred", "error", "History view doesn’t show correctness plot.");
  setBoxState("boxConf", "imgConf", "msgConf", "error", "History view doesn’t show confusion matrix.");
  await loadPNGInto("/api/mlp/plot_history.png", "boxHist", "imgHistory2", "msgHist");
}

function toggleGraphButtons(canTrainPlots, canTestPlots){
  const bT = $('btnPlotTrain');
  const bE = $('btnPlotTest');
  const bH = $('btnPlotHistory');
  if (bT) bT.disabled = !canTrainPlots;
  if (bE) bE.disabled = !canTestPlots;
  if (bH) bH.disabled = !canTrainPlots;
}

// -------------------------
// Init
// -------------------------
window.addEventListener('DOMContentLoaded', () => {
  // Actions
  $('btnCreate')?.addEventListener('click', onCreate);
  $('fileMLP')?.addEventListener('change', onPickMLPFile);
  $('btnApplyMLP')?.addEventListener('click', onApplyMLP);
  $('btnTrain')?.addEventListener('click', onTrain);
  $('btnContinue')?.addEventListener('click', onContinue);
  $('btnEvalOneMLP')?.addEventListener('click', onEvalOneMLP);
  $('btnEvalFileMLP')?.addEventListener('click', onEvalFileMLP);
  $('btnSave')?.addEventListener('click', onSave);

  // Graph buttons
  $('btnPlotTrain')?.addEventListener('click', onPlotTrain);
  $('btnPlotTest')?.addEventListener('click', onPlotTest);
  $('btnPlotHistory')?.addEventListener('click', onPlotHistory);

  // Estado inicial de gráficos y botones
  clearGraphs();
  toggleGraphButtons(false, false);
});