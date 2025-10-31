const $ = (id) => document.getElementById(id);
    const bust = (u) => u + (u.includes('?') ? '&' : '?') + '__t=' + Date.now();

    function showImg(id, url){
      const img = $(id);
      img.style.display = 'none';
      img.onload = () => img.style.display = 'block';
      img.onerror = () => img.style.display = 'none';
      img.src = bust(url);
    }

    async function api(path, payload){
      const res = await fetch(path, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload||{})
      });
      const txt = await res.text();
      let data = {};
      try { data = JSON.parse(txt); } catch(e){ data = {error:'Invalid response'}; }
      if(!res.ok) throw new Error(data.error || 'Request failed');
      return data;
    }

    function parseBounds(s){
      const [a,b] = (s || '').split(',').map(v=>parseFloat(v.trim()));
      if(Number.isFinite(a) && Number.isFinite(b) && a < b) return [a,b];
      return [-10,10];
    }

    function renderHistory(history){
      if(!Array.isArray(history) || !history.length){ $('historyTable').innerHTML = ''; return; }
      let html = '<table><thead><tr><th>Iter</th><th>Max</th><th>Median</th><th>Min</th></tr></thead><tbody>';
      for(const h of history){
        html += `<tr>
          <td>${h.iter}</td>
          <td>${h.max.toFixed(6)}</td>
          <td>${h.median.toFixed(6)}</td>
          <td>${h.min.toFixed(6)}</td>
        </tr>`;
      }
      html += '</tbody></table>';
      $('historyTable').innerHTML = html;
    }

    async function runGA(){
      $('status').textContent = 'Running...';
      $('historyTable').innerHTML = '';
      $('bestVars').textContent = '—';
      $('bestFit').textContent  = '—';
      $('bestIters').textContent = '—';

      try{
        const payload = {
          func: $('func').value,
          pop: parseInt($('pop').value,10) || 30,
          iters: parseInt($('iters').value,10) || 300,
          threshold: parseFloat($('threshold').value) || 0.001,
          mutvar: parseFloat($('mutvar').value) || 0.5,
          cr: parseFloat($('cr').value) || 0.9,
          mr: parseFloat($('mr').value) || 0.2,
          bounds: parseBounds($('bounds').value)
        };
        const data = await api('/api/ga/run', payload);
        $('status').textContent = 'Done.';
        renderHistory(data.history || []);
        $('bestVars').textContent = '[' + (data.best_vars || []).map(v=>v.toFixed(6)).join(', ') + ']';
        $('bestFit').textContent  = (data.best_fitness!=null) ? data.best_fitness.toFixed(6) : '—';
        $('bestIters').textContent = data.iterations || '—';
        showImg('imgHistory', '/api/ga/plot_history.png');
      }catch(e){
        $('status').textContent = 'Error: ' + e.message;
      }
    }

    function showHistory(){
      showImg('imgHistory', '/api/ga/plot_history.png');
    }
    function showSurface(){
      showImg('imgSurface', '/api/ga/plot_surface.png');
    }

    window.addEventListener('DOMContentLoaded', ()=>{
      $('btnRun').addEventListener('click', runGA);
      $('btnHistory').addEventListener('click', showHistory);
      $('btnSurface').addEventListener('click', showSurface);
    });