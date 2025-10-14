const $ = (id) => document.getElementById(id);

async function api(path, data) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data || {}),
  });
  const txt = await res.text();
  try {
    const json = JSON.parse(txt);
    if (!res.ok) throw new Error(json.error || "Request failed");
    return json;
  } catch (e) {
    throw new Error("Invalid response");
  }
}

// -------------------
// CREATE / LOAD
// -------------------
async function onCreate() {
  const payload = {
    n_hidden: parseInt($("nHidden").value),
    shape: $("shape").value,
    width: parseInt($("width").value),
    dropout: parseFloat($("dropout").value),
    epochs: parseInt($("epochs").value),
    lr: parseFloat($("lr").value),
    optimizer: $("optimizer").value,
    batch_size: parseInt($("batch").value),
    four_classes: $("fourClasses").value === "true",
  };
  try {
    const r = await api("/api/create", payload);
    $("statusCreate").textContent = "Red creada correctamente. " + r.summary;
  } catch (e) {
    $("statusCreate").textContent = "Error: " + e.message;
  }
}

async function onLoad() {
  try {
    const r = await api("/api/load", {});
    $("statusCreate").textContent = "Modelo cargado: " + r.summary;
  } catch (e) {
    $("statusCreate").textContent = "Error: " + e.message;
  }
}

// -------------------
// TRAIN / SAVE
// -------------------
async function onTrain() {
  $("trainLog").textContent = "Entrenando...";
  try {
    const r = await api("/api/train", {});
    $("trainLog").innerHTML = r.log_html;
  } catch (e) {
    $("trainLog").textContent = "Error: " + e.message;
  }
}

async function onSave() {
  try {
    const r = await api("/api/save", {});
    $("statusSave").textContent = "Modelo guardado: " + r.path;
  } catch (e) {
    $("statusSave").textContent = "Error: " + e.message;
  }
}

// -------------------
// PLOTS
// -------------------
function showImg(id, url) {
  const img = $(id);
  img.style.display = "none";
  const src = url + "?t=" + Date.now();
  img.onload = () => (img.style.display = "block");
  img.onerror = () => (img.style.display = "none");
  img.src = src;
}

function onLoss() {
  showImg("imgLoss", "/plot/loss.png");
}
function onConf() {
  showImg("imgConf", "/plot/conf.png");
}

// -------------------
window.addEventListener("DOMContentLoaded", () => {
  $("btnCreate").addEventListener("click", onCreate);
  $("btnLoad").addEventListener("click", onLoad);
  $("btnTrain").addEventListener("click", onTrain);
  $("btnSave").addEventListener("click", onSave);
  $("btnLoss").addEventListener("click", onLoss);
  $("btnConf").addEventListener("click", onConf);
});