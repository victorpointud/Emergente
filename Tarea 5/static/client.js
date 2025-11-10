async function refresh() {
  const r = await fetch("/api/grid");
  const j = await r.json();
  document.getElementById("grid").textContent = j.grid;
}
async function send(cmd) {
  const r = await fetch("/api/command",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({cmd})});
  const j = await r.json();
  document.getElementById("grid").textContent = j.grid || "";
  document.getElementById("msg").textContent = j.msg || "";
}
document.getElementById("send").onclick = ()=>{
  const el = document.getElementById("cmd");
  send(el.value); el.value="";
};
document.getElementById("tick").onclick = ()=> send("");
document.getElementById("cmd").addEventListener("keydown",(e)=>{
  if(e.key==="Enter"){ e.preventDefault(); const v=e.target.value; send(v); e.target.value=""; }
});
document.getElementById("saveBtn").onclick = async ()=>{
  const name = document.getElementById("saveName").value || "city";
  const r = await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});
  const j = await r.json(); document.getElementById("msg").textContent = j.msg || "";
};
document.getElementById("loadBtn").onclick = async ()=>{
  const name = document.getElementById("loadName").value || "city";
  const r = await fetch("/api/load",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});
  const j = await r.json();
  if(j.grid) document.getElementById("grid").textContent = j.grid;
  document.getElementById("msg").textContent = j.msg || "";
};
refresh();