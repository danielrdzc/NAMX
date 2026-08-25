#pragma once
//
// Presets: el estado completo del pedal, guardado en disco.
//
// Un preset guarda las tres cosas que definen el sonido:
//   - que modelo esta cargado
//   - en que orden estan los bloques
//   - el valor de cada perilla, incluido si el bloque esta encendido
//
// Sin el modelo adentro un preset esta a medias: cargarlo te dejaria las
// perillas de un sonido con el ampli de otro.
//
// Todo esto corre en el hilo de control. Cargar un preset mueve parametros
// (atomics), reordena la cadena (doble lista + atomico) y cambia el modelo
// (doble ranura + atomico). Ninguna de las tres cosas interrumpe el audio.
//
#include "effects.h"
#include "nam_effect.h"
#include "ir.h"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>

namespace fx {

class Presets {
public:
    Presets(Chain* chain, NamModel* nam, IRLoader* ir, std::filesystem::path dir)
        : _chain(chain), _nam(nam), _ir(ir), _dir(std::move(dir)) {
        std::error_code ec;
        std::filesystem::create_directories(_dir, ec);
    }

    std::vector<std::string> list() const {
        std::vector<std::string> names;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(_dir, ec)) {
            if (e.is_regular_file() && e.path().extension() == ".json")
                names.push_back(e.path().stem().string());
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    std::string save(const std::string& name) {
        if (!validName(name)) return "nombre invalido";

        nlohmann::json j;
        j["name"]  = name;
        j["model"] = _nam->path();
        j["ir"]    = _ir->path();
        j["order"] = _chain->order();

        nlohmann::json params = nlohmann::json::object();
        for (const auto& p : _chain->collectParams()) params[p.id] = p.get();
        j["params"] = params;

        std::ofstream f(_dir / (name + ".json"));
        if (!f) return "no se pudo escribir el archivo";
        f << j.dump(2) << "\n";
        return "";
    }

    std::string load(const std::string& name) {
        if (!validName(name)) return "nombre invalido";
        const std::filesystem::path file = _dir / (name + ".json");
        if (!std::filesystem::exists(file)) return "no existe el preset";

        nlohmann::json j;
        try {
            std::ifstream f(file);
            f >> j;
        } catch (std::exception& e) {
            return std::string("json invalido: ") + e.what();
        }

        // 1. El modelo primero: es lo mas lento y lo que mas cambia el sonido.
        if (j.contains("model") && j["model"].is_string()) {
            const std::string m = j["model"].get<std::string>();
            if (!m.empty() && m != _nam->path()) {
                const std::string err = _nam->loadModel(m);
                if (!err.empty()) return "modelo: " + err;
            }
        }

        // 1b. La IR: igual que el modelo, es parte del sonido base.
        if (j.contains("ir") && j["ir"].is_string()) {
            const std::string ir = j["ir"].get<std::string>();
            if (ir != _ir->path()) _ir->loadIR(ir);      // vacio = descargar
        }

        // 2. El orden, antes que los parametros: collectParams los devuelve en
        //    orden de senal, y queremos aplicar sobre la cadena ya acomodada.
        if (j.contains("order") && j["order"].is_array()) {
            std::vector<std::string> order;
            for (const auto& v : j["order"])
                if (v.is_string()) order.push_back(v.get<std::string>());
            _chain->setOrder(order);          // si no valida, se queda como estaba
        }

        // 3. Las perillas.
        if (j.contains("params") && j["params"].is_object()) {
            auto live = _chain->collectParams();
            for (auto& p : live) {
                auto it = j["params"].find(p.id);
                if (it != j["params"].end() && it->is_number())
                    p.set(it->get<float>());
            }
        }
        return "";
    }

    std::string remove(const std::string& name) {
        if (!validName(name)) return "nombre invalido";
        std::error_code ec;
        if (!std::filesystem::remove(_dir / (name + ".json"), ec))
            return "no se pudo borrar";
        return "";
    }

    // Modelos disponibles, para el selector de ampli.
    std::vector<std::string> models(const std::filesystem::path& modelsDir) const {
        return scan(modelsDir, ".nam");
    }
    std::vector<std::string> irs(const std::filesystem::path& irDir) const {
        return scan(irDir, ".wav");
    }

    static std::vector<std::string> scan(const std::filesystem::path& dir,
                                         const std::string& ext) {
        std::vector<std::string> out;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (!e.is_regular_file()) continue;
            std::string got = e.path().extension().string();
            for (auto& c : got) c = static_cast<char>(std::tolower(c));
            if (got == ext) out.push_back(e.path().string());
        }
        std::sort(out.begin(), out.end());
        return out;
    }

private:
    // Los nombres vienen de una peticion HTTP: no dejar que se salgan del
    // directorio con "../" ni metan separadores de ruta.
    static bool validName(const std::string& n) {
        if (n.empty() || n.size() > 64) return false;
        for (char c : n) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                         || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == ' ';
            if (!ok) return false;
        }
        return true;
    }

    Chain*    _chain;
    NamModel* _nam;
    IRLoader* _ir;
    std::filesystem::path _dir;
};

} // namespace fx
