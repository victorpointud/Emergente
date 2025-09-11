let mlpConfigText = null

const $ = (id) => document.getElementById(id)

function parseCSVLine(line){ return line.split(',').map(s=>s.trim()).filter(s=>s.length>0) }

function readLocalFile(file){
return new Promise((resolve, reject) => {
    if(!file) return reject(new Error("No file selected"))
    const r = new FileReader()
    r.onload = () => resolve(r.result)
    r.onerror = () => reject(new Error("Error reading file"))
    r.readAsText(file, 'utf-8')
})
}

async function api(path, payload){
const res = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)})
const txt = await res.text()
let data = {}
try{ data = JSON.parse(txt) } catch { data = {error:"Invalid response"} }
if(!res.ok) throw new Error(data.error || "Request failed")
return data
}

function parseHiddenNeurons(text, nHidden){
const raw = text.trim()
if(!raw) return Array(nHidden).fill(4)
if(raw.indexOf(',')>=0) return raw.split(',').map(s=>parseInt(s.trim(),10))
return Array(nHidden).fill(parseInt(raw,10))
}

function toRowsFromCSVText(t){
const lines = t.split(/\r?\n/).map(l=>l.trim()).filter(l=>l.length>0)
return lines.map(line => parseCSVLine(line).map(Number))
}

async function onCreate(){
const n_in = parseInt($('nIn').value,10)
const n_out = parseInt($('nOut').value,10)
const n_hidden = parseInt($('nHidden').value,10)
const hiddenNeurons = parseHiddenNeurons($('hiddenNeurons').value, n_hidden)
const resp = await api('/api/mlp/create', {n_in, n_out, n_hidden, hidden_neurons: hiddenNeurons})
$('createPreview').textContent = `Created. Sizes: [${resp.sizes.join(', ')}]`
}

function onPickMLPFile(){
const f = $('fileMLP').files[0] || null
mlpConfigText = null
$('mlpPreview').textContent = f ? 'File selected. Press “Apply configuration”.' : ''
if(f){ readLocalFile(f).then(t=> mlpConfigText=t).catch(e=> $('mlpPreview').textContent=`Error: ${e.message}`) }
}

async function onApplyMLP(){
if(!mlpConfigText){ $('mlpPreview').textContent='Select a configuration file.'; return }
try{
    const cfg = JSON.parse(mlpConfigText)
    await api('/api/mlp/apply_config', cfg)
    $('mlpPreview').textContent = `Applied. Sizes: [${cfg.sizes.join(', ')}]`
}catch(err){
    $('mlpPreview').textContent = `Error: ${err.message}`
}
}

async function onTrain(){
try{
    const act_hidden = $('actHidden').value
    const act_out = $('actOut').value
    const epochs = parseInt($('epochs').value,10)
    const lr = parseFloat($('lr').value)
    const XtrFile = $('fileXtr').files[0]; const YtrFile = $('fileYtr').files[0]
    if(!XtrFile || !YtrFile){ $('trainLog').textContent='Select Train X and Train Y CSVs.'; return }
    const Xtr = toRowsFromCSVText(await readLocalFile(XtrFile))
    const Ytr = toRowsFromCSVText(await readLocalFile(YtrFile))
    let Xte = [], Yte = []
    if($('fileXte').files[0] && $('fileYte').files[0]){
    Xte = toRowsFromCSVText(await readLocalFile($('fileXte').files[0]))
    Yte = toRowsFromCSVText(await readLocalFile($('fileYte').files[0]))
    }
    const resp = await api('/api/mlp/train', {X_train:Xtr, Y_train:Ytr, X_test:Xte, Y_test:Yte, epochs, lr, act_hidden, act_out})
    const hist = resp.history || []
    let html = '<table><thead><tr><th>Epoch</th><th>Train Acc</th><th>Test Acc</th></tr></thead><tbody>'
    for(const h of hist){
    const ta = (h.train_acc!=null)? h.train_acc.toFixed(4) : '—'
    const te = (h.test_acc!=null)? h.test_acc.toFixed(4) : '—'
    html += `<tr><td>${h.epoch}</td><td>${ta}</td><td>${te}</td></tr>`
    }
    html += '</tbody></table>'
    $('trainLog').innerHTML = html
}catch(e){
    $('trainLog').textContent = `Error: ${e.message}`
}
}

async function onContinue(){ return onTrain() }

function actPair(){ return {act_hidden:$('actHidden').value, act_out:$('actOut').value} }

function onEvalOneMLP(){
const xTxt = $('inputVec').value.trim()
if(!xTxt){ $('oneOutMLP').textContent='Enter a vector.'; return }
const x = parseCSVLine(xTxt).map(Number)
const {act_hidden,act_out} = actPair()
api('/api/mlp/eval_one', {x, act_hidden, act_out})
    .then(({y}) => $('oneOutMLP').textContent = `y(${act_hidden}→${act_out}) = [${y.map(v=>Number(v).toFixed(6)).join(', ')}]`)
    .catch(e => $('oneOutMLP').textContent = `Error: ${e.message}`)
}

async function onEvalFileMLP(){
const f = $('fileInputsMLP').files[0]
if(!f){ $('manyOutMLP').textContent='Select a CSV file.'; return }
try{
    const rows = toRowsFromCSVText(await readLocalFile(f))
    const {act_hidden,act_out} = actPair()
    const resp = await api('/api/mlp/eval_file', {rows, act_hidden, act_out})
    const items = resp.results || []
    let html = '<table><thead><tr><th>#</th><th>x</th><th>y</th><th>Status</th></tr></thead><tbody>'
    for(const it of items){
    const y = Array.isArray(it.y)? `[${it.y.map(v=>Number(v).toFixed(6)).join(', ')}]` : '—'
    html += `<tr><td>${it.idx}</td><td>[${(it.x||[]).join(', ')}]</td><td>${y}</td><td>${it.err?('Error: '+it.err):'OK'}</td></tr>`
    }
    html += '</tbody></table>'
    $('manyOutMLP').innerHTML = html
}catch(e){
    $('manyOutMLP').textContent = `Error: ${e.message}`
}
}

async function onSave(){
try{
    const obj = await api('/api/mlp/save', {})
    const blob = new Blob([JSON.stringify(obj)], {type:'application/json'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'mlp_saved.json'
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
}catch(e){
    alert('Error: '+e.message)
}
}

window.addEventListener('DOMContentLoaded', ()=>{
$('btnCreate').addEventListener('click', onCreate)
$('fileMLP').addEventListener('change', onPickMLPFile)
$('btnApplyMLP').addEventListener('click', onApplyMLP)
$('btnTrain').addEventListener('click', onTrain)
$('btnContinue').addEventListener('click', onContinue)
$('btnEvalOneMLP').addEventListener('click', onEvalOneMLP)
$('btnEvalFileMLP').addEventListener('click', onEvalFileMLP)
$('btnSave').addEventListener('click', onSave)
})