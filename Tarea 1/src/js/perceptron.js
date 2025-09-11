let bias = null
let weights = null
let pendingWeightsFile = null

const $ = (id) => document.getElementById(id)

function parseCSVLine(line){
return line.split(',').map(s=>s.trim()).filter(s=>s.length>0)
}

function readLocalFile(file){
return new Promise((resolve, reject) => {
if(!file) return reject(new Error("No file selected"))
const reader = new FileReader()
reader.onload = () => resolve(reader.result)
reader.onerror = () => reject(new Error("Error reading file"))
reader.readAsText(file, 'utf-8')
})
}

async function api(path, payload){
const res = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)})
const txt = await res.text()
let data = {}
try { data = JSON.parse(txt) } catch { data = {error:"Invalid response"} }
if(!res.ok) throw new Error(data.error || "Request failed")
return data
}

function onWeightsInputChange(){
pendingWeightsFile = $('fileWeights').files[0] || null
bias = null
weights = null
$('btnEvalOne').disabled = true
$('btnEvalMany').disabled = true
$('weightsPreview').textContent = pendingWeightsFile ? 'File selected. Press “Apply configuration”.' : ''
}

async function onApplyConfig(){
if(!pendingWeightsFile){
$('weightsPreview').textContent = 'Select a weights file.'
return
}
try{
const content = await readLocalFile(pendingWeightsFile)
const line = content.split(/\r?\n/).find(l=>l.trim().length>0)
if(!line) throw new Error('Weights file is empty.')
const parts = parseCSVLine(line)
const nums = parts.map(Number)
if(nums.some(isNaN)) throw new Error('Non-numeric values in weights file.')
if(nums.length<2) throw new Error('At least bias and one weight are required.')
const b = nums[0]
const w = nums.slice(1)
const r = await api("/api/apply_config", {bias:b, weights:w})
bias = b
weights = w
$('weightsPreview').textContent = `Bias = ${bias} | Weights = [${weights.join(', ')}] (n=${weights.length})`
$('btnEvalOne').disabled = false
$('btnEvalMany').disabled = false
}catch(err){
$('weightsPreview').textContent = `Error: ${err.message}`
bias = null
weights = null
$('btnEvalOne').disabled = true
$('btnEvalMany').disabled = true
}
}

function onEvalOne(){
if(bias===null || !Array.isArray(weights)){
$('oneResult').textContent = 'Apply configuration first.'
return
}
const txt = $('inputVector').value.trim()
if(!txt){
$('oneResult').textContent = 'Enter a vector (comma-separated values).'
return
}
const parts = parseCSVLine(txt)
const x = parts.map(Number)
if(x.length!==weights.length){
$('oneResult').textContent = `Expected ${weights.length} values. Received: ${x.length}.`
return
}
if(x.some(isNaN)){
$('oneResult').textContent = 'The vector contains non-numeric values.'
return
}
const act = $('activation').value
api("/api/eval_one", {x, activation: act})
.then(({z,y}) => { $('oneResult').textContent = `x=[${x.join(', ')}] → z=${z.toFixed(6)}, y(${act})=${y.toFixed(6)}` })
.catch(e => { $('oneResult').textContent = `Error: ${e.message}` })
}

async function onEvalFile(){
if(bias===null || !Array.isArray(weights)){
$('manyResults').textContent = 'Apply configuration first.'
return
}
const file = $('fileInputs').files[0]
if(!file){
$('manyResults').textContent = 'Select an inputs file (.csv).'
return
}
try{
const act = $('activation').value
const content = await readLocalFile(file)
const lines = content.split(/\r?\n/).map(l=>l.trim()).filter(l=>l.length>0)
if(lines.length===0){
    $('manyResults').textContent = 'Inputs file is empty.'
    return
}
const rows = []
for(const line of lines){
    const parts = parseCSVLine(line)
    rows.push(parts.map(Number))
}
const resp = await api("/api/eval_file", {rows, activation: act})
const filas = resp.results || []
let html = '<table><thead><tr><th>#</th><th>x</th><th>z</th><th>y(' + act + ')</th><th>Status</th></tr></thead><tbody>'
for(const f of filas){
    const z = (f.z===null || f.z===undefined) ? '—' : Number(f.z).toFixed(6)
    const y = (f.y===null || f.y===undefined) ? '—' : Number(f.y).toFixed(6)
    html += `<tr><td>${f.idx}</td><td>[${(f.x||[]).join(', ')}]</td><td>${z}</td><td>${y}</td><td>${f.err?('Error: '+f.err):'OK'}</td></tr>`
}
html += '</tbody></table>'
$('manyResults').innerHTML = html
}catch(err){
$('manyResults').textContent = `Error: ${err.message}`
}
}

window.addEventListener('DOMContentLoaded',()=>{
$('fileWeights').addEventListener('change', onWeightsInputChange)
$('btnLoadConfig').addEventListener('click', onApplyConfig)
$('btnEvalOne').addEventListener('click', onEvalOne)
$('btnEvalMany').addEventListener('click', onEvalFile)
})