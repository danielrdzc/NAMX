#pragma once
//
// Interfaz web del pedal.
//
// El pedal sirve su propia pagina. Abres http://nampedal.local:8080 desde
// Windows, desde el telefono, desde lo que sea: no hay nada que instalar del
// otro lado. Es el mismo modelo que usan MOD Devices y Zynthian.
//
// Seguridad de tiempo real: este hilo SOLO llama a los setters de los
// parametros, que escriben atomics. Nunca toca el hilo de audio ni sus buffers,
// y no hay un solo candado compartido entre los dos.
//
// Sin dependencias externas: un servidor HTTP minimo hecho a mano. Solo necesita
// GET, y responde una peticion por conexion.
//
#include "params.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
#endif

namespace fx {

class WebUI {
public:
    ~WebUI() { stop(); }

#if defined(_WIN32)
    bool start(int, std::vector<Param>*) {
        std::printf("Web UI: no disponible en la compilacion de Windows.\n");
        return false;
    }
    void stop() {}
#else

    bool start(int port, std::vector<Param>* params) {
        _params = params;
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
            handle(c);
            ::close(c);
        }
    }

    void handle(int c) {
        char buf[4096];
        const ssize_t n = ::recv(c, buf, sizeof(buf) - 1, 0);
        if (n <= 0) return;
        buf[n] = '\0';

        // "GET /ruta?consulta HTTP/1.1"
        const char* sp1 = std::strchr(buf, ' ');
        if (!sp1) return;
        const char* sp2 = std::strchr(sp1 + 1, ' ');
        if (!sp2) return;
        std::string target(sp1 + 1, static_cast<size_t>(sp2 - sp1 - 1));

        std::string path = target, query;
        const size_t q = target.find('?');
        if (q != std::string::npos) { path = target.substr(0, q); query = target.substr(q + 1); }

        if (path == "/")                 { send(c, "text/html",        html()); }
        else if (path == "/api/params")  { send(c, "application/json", paramsJson()); }
        else if (path == "/api/set")     { send(c, "application/json", setParam(query)); }
        else                             { sendStatus(c, "404 Not Found", "text/plain", "no"); }
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
        for (auto& p : *_params) {
            if (p.id == id) { p.set(v); return "{\"ok\":true}"; }
        }
        return "{\"ok\":false}";
    }

    std::string paramsJson() {
        std::string j = "[";
        for (size_t i = 0; i < _params->size(); ++i) {
            const Param& p = (*_params)[i];
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
    std::vector<Param>* _params = nullptr;
#endif
};

#if !defined(_WIN32)
inline std::string WebUI::html() {
    return R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMX</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; padding:20px; background:#111; color:#eee;
         font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; }
  h1 { font-size:18px; margin:0 0 20px; letter-spacing:.08em; color:#888;
       text-transform:uppercase; }
  .fx { background:#1b1b1b; border:1px solid #2a2a2a; border-radius:10px;
        padding:14px 16px; margin-bottom:12px; }
  .fx.off { opacity:.45; }
  .hd { display:flex; align-items:center; justify-content:space-between;
        margin-bottom:10px; }
  .nm { font-weight:600; text-transform:capitalize; letter-spacing:.03em; }
  .row { display:grid; grid-template-columns:90px 1fr 68px; gap:12px;
         align-items:center; margin:9px 0; }
  .lb { color:#999; font-size:13px; }
  .vl { color:#5ac8fa; font-variant-numeric:tabular-nums; text-align:right;
        font-size:13px; }
  input[type=range] { width:100%; accent-color:#5ac8fa; }
  button { background:#2a2a2a; color:#ddd; border:1px solid #3a3a3a;
           border-radius:999px; padding:5px 14px; font-size:13px; cursor:pointer; }
  button.on { background:#5ac8fa; border-color:#5ac8fa; color:#062028;
              font-weight:600; }
</style>
<h1>NAMX</h1>
<div id="app"></div>
<script>
const app = document.getElementById('app');
let groups = {};

function setParam(id, v) {
  fetch('/api/set?id=' + encodeURIComponent(id) + '&value=' + v);
}

function build(params) {
  groups = {};
  for (const p of params) {
    const g = p.id.split('.')[0];
    (groups[g] = groups[g] || []).push(p);
  }
  app.innerHTML = '';
  for (const [name, list] of Object.entries(groups)) {
    const box = document.createElement('div');
    box.className = 'fx';
    const on = list.find(p => p.id.endsWith('.enabled'));

    const hd = document.createElement('div');
    hd.className = 'hd';
    hd.innerHTML = '<span class="nm">' + name + '</span>';
    if (on) {
      const b = document.createElement('button');
      const paint = () => {
        b.textContent = on.value >= 0.5 ? 'ON' : 'OFF';
        b.className = on.value >= 0.5 ? 'on' : '';
        box.className = 'fx' + (on.value >= 0.5 ? '' : ' off');
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
    app.appendChild(box);
  }
}

fetch('/api/params').then(r => r.json()).then(build)
  .catch(() => app.textContent = 'No se pudo hablar con el pedal.');
</script>
)HTML";
}
#endif

} // namespace fx
