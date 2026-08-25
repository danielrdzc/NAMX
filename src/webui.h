#pragma once
//
// Interfaz web del pedal.
//
// El pedal sirve su propia pagina. Abres http://nampedal.local:8080 desde
// Windows, desde el telefono, desde lo que sea: no hay nada que instalar del
// otro lado. Es el mismo modelo que usan MOD Devices y Zynthian.
//
// Seguridad de tiempo real: este hilo SOLO llama a setters de parametros (que
// escriben atomics) y a Chain::setOrder (que usa doble lista + atomico). Nunca
// toca los buffers del hilo de audio ni comparte un candado con el.
//
// Sin dependencias externas: un servidor HTTP minimo hecho a mano.
//
#include "effects.h"
#include "nam_effect.h"
#include "params.h"
#include "presets.h"
#include "tuner.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <sys/time.h>
  #include <unistd.h>
#endif

namespace fx {

class WebUI {
public:
    ~WebUI() { stop(); }

#if defined(_WIN32)
    bool start(int, Chain*, Presets*, NamModel*, IRLoader*, Tuner*,
               const std::string&, const std::string&) {
        std::printf("Web UI: no disponible en la compilacion de Windows.\n");
        return false;
    }
    void stop() {}
#else

    bool start(int port, Chain* chain, Presets* presets, NamModel* nam,
               IRLoader* ir, Tuner* tuner,
               const std::string& modelsDir, const std::string& irDir) {
        _chain = chain; _presets = presets; _nam = nam; _ir = ir; _tuner = tuner;
        _modelsDir = modelsDir; _irDir = irDir;
        _fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (_fd < 0) return false;

        int yes = 1;
        ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(static_cast<uint16_t>(port));
        if (::bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(_fd); _fd = -1; return false;
        }
        if (::listen(_fd, 8) < 0) { ::close(_fd); _fd = -1; return false; }

        _run.store(true);
        _th = std::thread([this] { loop(); });
        return true;
    }

    void stop() {
        if (!_run.exchange(false)) return;
        if (_fd >= 0) { ::shutdown(_fd, SHUT_RDWR); ::close(_fd); _fd = -1; }
        if (_th.joinable()) _th.join();
    }

private:
    void loop() {
        while (_run.load()) {
            int c = ::accept(_fd, nullptr, nullptr);
            if (c < 0) { if (!_run.load()) break; continue; }

            // Chrome abre conexiones especulativas que no mandan nada nunca.
            // Sin este timeout, recv() se bloquea para siempre en una de esas y
            // el servidor entero deja de atender.
            timeval tv{};
            tv.tv_sec = 2;
            ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            ::setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

            handle(c);
            ::close(c);
        }
    }

    void handle(int c) {
        char buf[8192];
        const ssize_t n = ::recv(c, buf, sizeof(buf) - 1, 0);
        if (n <= 0) return;
        buf[n] = '\0';

        const char* sp1 = std::strchr(buf, ' ');
        if (!sp1) return;
        const char* sp2 = std::strchr(sp1 + 1, ' ');
        if (!sp2) return;
        std::string target(sp1 + 1, static_cast<size_t>(sp2 - sp1 - 1));

        std::string path = target, query;
        const size_t q = target.find('?');
        if (q != std::string::npos) { path = target.substr(0, q); query = target.substr(q + 1); }

        if      (path == "/")            send(c, "text/html",        html());
        else if (path == "/api/params")  send(c, "application/json", paramsJson());
        else if (path == "/api/set")     send(c, "application/json", setParam(query));
        else if (path == "/api/order")   send(c, "application/json", setOrder(query));
        else if (path == "/api/state")   send(c, "application/json", stateJson());
        else if (path == "/api/preset/save")   send(c, "application/json",
                                                    result(_presets->save(urlDecode(field(query,"name")))));
        else if (path == "/api/preset/load")   send(c, "application/json",
                                                    result(_presets->load(urlDecode(field(query,"name")))));
        else if (path == "/api/preset/delete") send(c, "application/json",
                                                    result(_presets->remove(urlDecode(field(query,"name")))));
        else if (path == "/api/model/load")    send(c, "application/json",
                                                    result(_nam->loadModel(urlDecode(field(query,"path")))));
        else if (path == "/api/ir/load")       send(c, "application/json",
                                                    result(_ir->loadIR(urlDecode(field(query,"path")))));
        else if (path == "/api/tuner")         send(c, "application/json", tunerJson());
        else                             sendStatus(c, "404 Not Found", "text/plain", "no");
    }

    static std::string urlDecode(const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i] == '+') { out.push_back(' '); }
            else if (in[i] == '%' && i + 2 < in.size()) {
                const std::string hex = in.substr(i + 1, 2);
                out.push_back(static_cast<char>(std::strtol(hex.c_str(), nullptr, 16)));
                i += 2;
            } else out.push_back(in[i]);
        }
        return out;
    }

    static std::string jsonEscape(const std::string& in) {
        std::string out;
        for (char c : in) {
            if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
            else out.push_back(c);
        }
        return out;
    }

    static std::string result(const std::string& err) {
        return err.empty() ? "{\"ok\":true}"
                           : "{\"ok\":false,\"error\":\"" + jsonEscape(err) + "\"}";
    }

    std::string tunerJson() {
        const TunerReading t = _tuner->detect();
        char b[256];
        std::snprintf(b, sizeof(b),
            "{\"valid\":%s,\"freq\":%.2f,\"cents\":%.1f,\"clarity\":%.2f,\"note\":\"%s\"}",
            t.valid ? "true" : "false", t.freq, t.cents, t.clarity, t.note.c_str());
        return b;
    }

    std::string stateJson() {
        std::string j = "{\"model\":\"" + jsonEscape(_nam->path()) + "\",\"models\":[";
        const auto ms = _presets->models(_modelsDir);
        for (size_t i = 0; i < ms.size(); ++i)
            j += (i ? "," : "") + std::string("\"") + jsonEscape(ms[i]) + "\"";
        j += "],\"ir\":\"" + jsonEscape(_ir->path()) + "\",\"irs\":[";
        const auto is = _presets->irs(_irDir);
        for (size_t i = 0; i < is.size(); ++i)
            j += (i ? "," : "") + std::string("\"") + jsonEscape(is[i]) + "\"";
        j += "],\"presets\":[";
        const auto ps = _presets->list();
        for (size_t i = 0; i < ps.size(); ++i)
            j += (i ? "," : "") + std::string("\"") + jsonEscape(ps[i]) + "\"";
        return j + "]}";
    }

    static std::string field(const std::string& query, const std::string& key) {
        const std::string k = key + "=";
        size_t p = query.find(k);
        while (p != std::string::npos && p != 0 && query[p - 1] != '&') p = query.find(k, p + 1);
        if (p == std::string::npos) return "";
        const size_t start = p + k.size();
        const size_t end   = query.find('&', start);
        return query.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }

    std::string setParam(const std::string& query) {
        const std::string id = field(query, "id");
        const std::string vs = field(query, "value");
        if (id.empty() || vs.empty()) return "{\"ok\":false}";
        const float v = static_cast<float>(std::atof(vs.c_str()));
        auto params = _chain->collectParams();
        for (auto& p : params)
            if (p.id == id) { p.set(v); return "{\"ok\":true}"; }
        return "{\"ok\":false}";
    }

    std::string setOrder(const std::string& query) {
        const std::string ids = field(query, "ids");
        if (ids.empty()) return "{\"ok\":false}";
        std::vector<std::string> names;
        size_t start = 0;
        while (start <= ids.size()) {
            const size_t comma = ids.find(',', start);
            const std::string one = ids.substr(start, comma == std::string::npos
                                                     ? std::string::npos : comma - start);
            if (!one.empty()) names.push_back(one);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return _chain->setOrder(names) ? "{\"ok\":true}" : "{\"ok\":false}";
    }

    std::string paramsJson() {
        auto params = _chain->collectParams();
        std::string j = "[";
        for (size_t i = 0; i < params.size(); ++i) {
            const Param& p = params[i];
            char tmp[512];
            std::snprintf(tmp, sizeof(tmp),
                "%s{\"id\":\"%s\",\"label\":\"%s\",\"unit\":\"%s\","
                "\"min\":%g,\"max\":%g,\"step\":%g,\"value\":%g}",
                i ? "," : "", p.id.c_str(), p.label.c_str(), p.unit.c_str(),
                p.min, p.max, p.step, p.get());
            j += tmp;
        }
        return j + "]";
    }

    void send(int c, const char* type, const std::string& body) {
        sendStatus(c, "200 OK", type, body);
    }

    void sendStatus(int c, const char* status, const char* type, const std::string& body) {
        char head[256];
        const int hn = std::snprintf(head, sizeof(head),
            "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
            status, type, body.size());
        ::send(c, head, static_cast<size_t>(hn), 0);
        ::send(c, body.data(), body.size(), 0);
    }

    static std::string html();

    int _fd = -1;
    std::atomic<bool> _run{false};
    std::thread _th;
    Chain*    _chain   = nullptr;
    Presets*  _presets = nullptr;
    NamModel* _nam     = nullptr;
    IRLoader* _ir      = nullptr;
    Tuner*    _tuner   = nullptr;
    std::string _modelsDir, _irDir;
#endif
};

#if !defined(_WIN32)
inline std::string WebUI::html() {
    return R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMX</title>
<style>
  :root { color-scheme: dark; --ac:#5ac8fa; }
  * { box-sizing: border-box; }
  body { margin:0; padding:22px 18px 60px; background:#0e0e10; color:#eee;
         font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;
         max-width:560px; margin-inline:auto; }
  h1 { font-size:17px; margin:0 0 4px; letter-spacing:.14em; color:#ddd; }
  .sub { color:#666; font-size:12.5px; margin-bottom:22px; }
  .end { text-align:center; color:#666; font-size:11.5px; letter-spacing:.16em;
         padding:7px 0; }
  .fx { background:#191a1d; border:1px solid #26272b; border-radius:11px;
        padding:12px 14px 10px; }
  .fx.off { opacity:.42; }
  .fx.drag { opacity:.35; border-color:var(--ac); }
  .fx.over { border-color:var(--ac); }
  .arrow { text-align:center; color:#3a3b40; font-size:13px; line-height:1;
           padding:5px 0; }
  .hd { display:flex; align-items:center; gap:10px; }
  .grip { cursor:grab; color:#54565c; font-size:15px; letter-spacing:-2px;
          user-select:none; padding:2px 3px; }
  .grip:active { cursor:grabbing; }
  .nm { font-weight:600; text-transform:uppercase; letter-spacing:.07em;
        font-size:12.5px; flex:1; }
  .row { display:grid; grid-template-columns:78px 1fr 66px; gap:11px;
         align-items:center; margin:8px 0 0; }
  .lb { color:#8a8c92; font-size:12.5px; }
  .vl { color:var(--ac); font-variant-numeric:tabular-nums; text-align:right;
        font-size:12.5px; }
  input[type=range] { width:100%; accent-color:var(--ac); }
  button { background:#26272b; color:#c8cad0; border:1px solid #34353a;
           border-radius:999px; padding:3px 12px; font-size:11.5px;
           cursor:pointer; letter-spacing:.05em; }
  button.on { background:var(--ac); border-color:var(--ac); color:#06222c;
              font-weight:700; }
  .bar { background:#191a1d; border:1px solid #26272b; border-radius:11px;
         padding:12px 14px; margin-bottom:18px; }
  .bl { display:block; color:#8a8c92; font-size:11.5px; letter-spacing:.07em;
        text-transform:uppercase; margin:0 0 5px; }
  .bar select { width:100%; background:#0e0e10; color:#ddd; border:1px solid #34353a;
                border-radius:7px; padding:6px 8px; font-size:13px; margin-bottom:12px; }
  .prow { display:flex; gap:7px; align-items:center; }
  .prow select { flex:1; margin-bottom:0; }
  .msg { color:#8a8c92; font-size:12px; min-height:16px; margin-top:9px; }
  .msg.err { color:#ff6b6b; }
  .tuner { background:#191a1d; border:1px solid #26272b; border-radius:11px;
           padding:14px; margin-bottom:18px; text-align:center; }
  .tnote { font-size:34px; font-weight:700; letter-spacing:.02em; line-height:1.1;
           color:#5a5c62; font-variant-numeric:tabular-nums; }
  .tnote.ok { color:#4ade80; }
  .tbar { position:relative; height:8px; background:#0e0e10; border-radius:999px;
          margin:10px 0 7px; overflow:hidden; }
  .tcenter { position:absolute; left:50%; top:0; width:2px; height:100%;
             background:#3a3b40; transform:translateX(-1px); }
  .tneedle { position:absolute; top:0; width:4px; height:100%; border-radius:2px;
             background:var(--ac); left:50%; transform:translateX(-2px);
             transition:left .08s linear; }
  .tinfo { color:#8a8c92; font-size:11.5px; letter-spacing:.05em; }
</style>
<h1>NAMX</h1>
<div class="sub">Arrastra los bloques por el asa para reordenar la cadena.</div>

<div class="bar">
  <label class="bl">Ampli</label>
  <select id="model"></select>
  <label class="bl">Gabinete (IR)</label>
  <select id="ir"></select>
  <label class="bl">Preset</label>
  <div class="prow">
    <select id="preset"></select>
    <button id="bload">Cargar</button>
    <button id="bsave">Guardar</button>
  </div>
  <div id="msg" class="msg"></div>
</div>

<div class="tuner" id="tuner">
  <div class="tnote" id="tnote">--</div>
  <div class="tbar"><div class="tneedle" id="tneedle"></div><div class="tcenter"></div></div>
  <div class="tinfo" id="tinfo">afinador</div>
</div>

<div class="end">GUITARRA</div>
<div id="app"></div>
<div class="end">SALIDA</div>
<script>
const app = document.getElementById('app');
let dragged = null;

const setParam = (id, v) =>
  fetch('/api/set?id=' + encodeURIComponent(id) + '&value=' + v);

function sendOrder() {
  const ids = [...app.querySelectorAll('.fx')].map(el => el.dataset.name);
  fetch('/api/order?ids=' + ids.join(','));
}

// Devuelve el bloque delante del cual hay que insertar, segun la altura del cursor.
function afterElement(y) {
  const els = [...app.querySelectorAll('.fx:not(.drag)')];
  let best = { off: -Infinity, el: null };
  for (const el of els) {
    const box = el.getBoundingClientRect();
    const off = y - box.top - box.height / 2;
    if (off < 0 && off > best.off) best = { off, el };
  }
  return best.el;
}

app.addEventListener('dragover', e => {
  e.preventDefault();
  if (!dragged) return;
  const after = afterElement(e.clientY);
  if (after == null) app.appendChild(dragged);
  else if (after !== dragged) app.insertBefore(dragged, after);
  redrawArrows();
});

function redrawArrows() {
  [...app.querySelectorAll('.arrow')].forEach(a => a.remove());
  const cards = [...app.querySelectorAll('.fx')];
  cards.forEach((c, i) => {
    if (i < cards.length - 1) {
      const a = document.createElement('div');
      a.className = 'arrow';
      a.textContent = '▼';
      c.after(a);
    }
  });
}

function build(params) {
  const groups = {};
  const order = [];
  for (const p of params) {
    const g = p.id.split('.')[0];
    if (!groups[g]) { groups[g] = []; order.push(g); }
    groups[g].push(p);
  }

  app.innerHTML = '';
  for (const name of order) {
    const list = groups[name];
    const box = document.createElement('div');
    box.className = 'fx';
    box.dataset.name = name;

    const hd = document.createElement('div');
    hd.className = 'hd';

    const grip = document.createElement('span');
    grip.className = 'grip';
    grip.textContent = '⠿';
    grip.title = 'Arrastra para reordenar';
    // Solo se puede arrastrar desde el asa: asi los sliders siguen usables.
    grip.addEventListener('mousedown', () => box.draggable = true);
    grip.addEventListener('touchstart', () => box.draggable = true, {passive:true});

    const nm = document.createElement('span');
    nm.className = 'nm';
    nm.textContent = name;

    hd.append(grip, nm);

    const on = list.find(p => p.id.endsWith('.enabled'));
    if (on) {
      const b = document.createElement('button');
      const paint = () => {
        const v = on.value >= 0.5;
        b.textContent = v ? 'ON' : 'OFF';
        b.className = v ? 'on' : '';
        box.classList.toggle('off', !v);
      };
      b.onclick = () => { on.value = on.value >= 0.5 ? 0 : 1; setParam(on.id, on.value); paint(); };
      paint();
      hd.appendChild(b);
    }
    box.appendChild(hd);

    for (const p of list) {
      if (p.id.endsWith('.enabled')) continue;
      const row = document.createElement('div');
      row.className = 'row';
      const lb = document.createElement('div');
      lb.className = 'lb'; lb.textContent = p.label;
      const sl = document.createElement('input');
      sl.type = 'range'; sl.min = p.min; sl.max = p.max; sl.step = p.step; sl.value = p.value;
      const vl = document.createElement('div');
      vl.className = 'vl';
      const show = v => vl.textContent = (+v).toFixed(p.step >= 1 ? 0 : 2) + ' ' + p.unit;
      show(p.value);
      sl.oninput = () => { show(sl.value); setParam(p.id, sl.value); };
      row.append(lb, sl, vl);
      box.appendChild(row);
    }

    box.addEventListener('dragstart', () => { dragged = box; box.classList.add('drag'); });
    box.addEventListener('dragend', () => {
      box.classList.remove('drag');
      box.draggable = false;
      dragged = null;
      sendOrder();
    });

    app.appendChild(box);
  }
  redrawArrows();
}

const msg = document.getElementById('msg');
const elModel = document.getElementById('model');
const elPreset = document.getElementById('preset');
const elIr = document.getElementById('ir');

function say(text, isErr) {
  msg.textContent = text;
  msg.className = 'msg' + (isErr ? ' err' : '');
  if (text) setTimeout(() => { if (msg.textContent === text) msg.textContent = ''; }, 3000);
}

const shortName = p => p.split('/').pop().replace(/\.nam$/i, '');

function fill(sel, items, current) {
  sel.innerHTML = '';
  for (const it of items) {
    const o = document.createElement('option');
    o.value = it;
    o.textContent = sel === elModel ? shortName(it) : it;
    if (it === current) o.selected = true;
    sel.appendChild(o);
  }
}

function refreshState() {
  return fetch('/api/state').then(r => r.json()).then(st => {
    fill(elModel, st.models, st.model);
    fill(elPreset, st.presets, null);
    // La opcion vacia permite quitar la IR: los captures "amp_cab" ya traen
    // gabinete y ponerles una IR encima seria apilar dos.
    fill(elIr, [''].concat(st.irs), st.ir);
    if (elIr.options.length) elIr.options[0].textContent = '(ninguno)';
  });
}

const refreshParams = () =>
  fetch('/api/params').then(r => r.json()).then(build);

elModel.onchange = () => {
  say('Cargando ampli...');
  fetch('/api/model/load?path=' + encodeURIComponent(elModel.value))
    .then(r => r.json())
    .then(r => { say(r.ok ? 'Ampli cargado' : r.error, !r.ok); });
};

elIr.onchange = () => {
  say('Cargando IR...');
  fetch('/api/ir/load?path=' + encodeURIComponent(elIr.value))
    .then(r => r.json())
    .then(r => { say(r.ok ? (elIr.value ? 'IR cargada' : 'IR quitada') : r.error, !r.ok);
                 refreshParams(); });
};

const tnote = document.getElementById('tnote');
const tneedle = document.getElementById('tneedle');
const tinfo = document.getElementById('tinfo');

setInterval(() => {
  fetch('/api/tuner').then(r => r.json()).then(t => {
    if (!t.valid) {
      tnote.textContent = '--';
      tnote.className = 'tnote';
      tinfo.textContent = 'toca una cuerda al aire';
      tneedle.style.left = '50%';
      return;
    }
    tnote.textContent = t.note;
    // +-50 cents ocupan todo el ancho de la barra
    const pos = Math.max(-50, Math.min(50, t.cents));
    tneedle.style.left = (50 + pos) + '%';
    const inTune = Math.abs(t.cents) < 5;
    tnote.className = 'tnote' + (inTune ? ' ok' : '');
    tinfo.textContent = t.freq.toFixed(1) + ' Hz   ' +
                        (t.cents >= 0 ? '+' : '') + t.cents.toFixed(0) + ' cents';
  }).catch(() => {});
}, 250);

document.getElementById('bload').onclick = () => {
  if (!elPreset.value) return;
  fetch('/api/preset/load?name=' + encodeURIComponent(elPreset.value))
    .then(r => r.json())
    .then(r => {
      say(r.ok ? 'Preset cargado' : r.error, !r.ok);
      if (r.ok) { refreshState(); refreshParams(); }
    });
};

document.getElementById('bsave').onclick = () => {
  const name = prompt('Nombre del preset:', elPreset.value || '');
  if (!name) return;
  fetch('/api/preset/save?name=' + encodeURIComponent(name))
    .then(r => r.json())
    .then(r => { say(r.ok ? 'Guardado' : r.error, !r.ok); if (r.ok) refreshState(); });
};

refreshState();
refreshParams().catch(() => app.textContent = 'No se pudo hablar con el pedal.');
</script>
)HTML";
}
#endif

} // namespace fx
