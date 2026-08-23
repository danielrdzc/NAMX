# NAM Pedal — puesta en marcha en Raspberry Pi 5

Hardware de esta etapa: Pi 5 + active cooler, Scarlett 2i2 por USB, protoboard con
botones y OLED SSD1306 128x64 por I2C.

Meta de la etapa: **el mismo binario que ya corre en Windows, corriendo en el Pi**,
con números medidos de xruns y latencia. La UI (OLED + botones) viene después;
primero audio, porque es lo que decide si el proyecto es viable.

---

## Etapa 1 — Flashear el SO (headless)

Usa **Raspberry Pi Imager** desde tu PC.

- SO: **Raspberry Pi OS Lite (64-bit)**. Lite, sin escritorio: menos procesos
  compitiendo con el hilo de audio, arranque más rápido, y de todos modos vas a
  entrar por SSH. Un pedal no necesita GNOME.
- Antes de escribir, abre **"Edit settings"** (el engrane / Ctrl+Shift+X) y llena:
  - hostname: `nampedal`
  - usuario y contraseña
  - Wi-Fi (SSID + password + país)
  - **Enable SSH** con autenticación por contraseña o tu llave pública

Arranca el Pi y entra:

```bash
ssh usuario@nampedal.local
```

**Alimentación:** usa la fuente oficial de 5V/5A. Con una fuente más débil el Pi 5
limita la corriente total de los puertos USB, y el 2i2 se alimenta por bus — puede
no enumerar o portarse errático. Si algo raro pasa con el audio, esto es lo primero
que hay que descartar.

---

## Etapa 2 — Sistema y dependencias

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git cmake build-essential pkg-config \
                    libasound2-dev i2c-tools
```

### RtAudio: ojo con la versión

Tu código usa la **API de RtAudio 6** (`getDeviceIds()`, ids en vez de índices).
Si el paquete de tu Pi OS trae RtAudio 5.x, **no compila**. Revisa:

```bash
sudo apt install -y librtaudio-dev
pkg-config --modversion rtaudio
```

- Si dice **6.x** → listo, sigue a la etapa 3.
- Si dice **5.x** (o no existe el paquete) → compílalo de fuente:

```bash
sudo apt remove -y librtaudio-dev
cd ~
git clone https://github.com/thestk/rtaudio.git
cd rtaudio && git checkout 6.0.1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRTAUDIO_API_ALSA=ON -DRTAUDIO_API_JACK=OFF
cmake --build build -j4
sudo cmake --install build
sudo ldconfig
```

---

## Etapa 3 — Traer el código y compilar

Lo más limpio es git, para que las dos máquinas queden sincronizadas. En Windows,
dentro de `C:\dev\nam-pedal-proto`:

```powershell
git init
git add .
git commit -m "prototipo NAM funcionando en Windows"
# crea el repo en GitHub y luego:
git remote add origin <tu-url>
git push -u origin main
```

En el Pi:

```bash
git clone <tu-url> ~/nam-pedal-proto
cd ~/nam-pedal-proto
cmake -B build
cmake --build build -j4
```

El `CMakeLists.txt` ya está preparado para esto: no encuentra vcpkg, cae a
pkg-config para RtAudio, y como Ninja/Make son de una sola configuración, fuerza
`CMAKE_BUILD_TYPE=Release` solo. **Eso importa muchísimo** — sin optimización la
inferencia de NAM es varias veces más lenta y vas a ver xruns que no son reales.

`build.ps1` es solo de Windows; en el Pi usa los dos comandos de cmake de arriba.

---

## Etapa 4 — Audio: primero ALSA crudo, luego el binario

Conecta el 2i2 a un puerto **USB 3** (los azules) y revisa que el sistema lo vea:

```bash
arecord -l     # tarjetas de captura
aplay -l       # tarjetas de salida
```

Anota el número de tarjeta (algo como `card 1: USB [Scarlett 2i2 USB]`).

Prueba fuera de tu programa, para separar problemas de driver de problemas de código:

```bash
arecord -D hw:1,0 -f S32_LE -r 48000 -c 2 -d 5 prueba.wav
aplay   -D hw:1,0 prueba.wav
```

Si eso funciona, corre lo tuyo. El programa ahora busca "Focusrite" o "Scarlett"
por nombre sin importar mayúsculas, así que debería auto-seleccionarlo:

```bash
./build/passthrough --list                      # ver los ids
./build/passthrough models/5150-blue-fullrig.nam --buffer 128
./build/passthrough --device 3 --buffer 128     # si la auto-detección falla
```

### Permisos de tiempo real

Tu código pide `RTAUDIO_SCHEDULE_REALTIME` con prioridad 90. En Linux eso **falla
en silencio** si el usuario no tiene permiso, y lo notas como xruns que no
deberían estar ahí:

```bash
sudo usermod -aG audio $USER
sudo tee /etc/security/limits.d/95-audio.conf <<'EOF'
@audio - rtprio 95
@audio - memlock unlimited
@audio - nice -19
EOF
```

Cierra la sesión SSH y vuelve a entrar (los límites se aplican al hacer login).
Verifica con `ulimit -r` → debe decir 95.

### Gobernador de CPU

```bash
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

Y revisa que el cooler esté haciendo su trabajo:

```bash
vcgencmd measure_temp
vcgencmd get_throttled     # 0x0 = sin throttling
```

### Qué medir

Corre el binario a 256, 128, 64 frames y anota, para cada uno:

| buffer | xruns en 5 min | latencia que reporta | ¿suena limpio? |
|--------|----------------|----------------------|----------------|
| 256    |                |                      |                |
| 128    |                |                      |                |
| 64     |                |                      |                |

Ese es el número que decide todo lo demás. El 2i2 por USB va a ser tu piso; un
HAT I2S después debería mejorarlo.

---

## Etapa 5 — Afinado de tiempo real (cuando ya tengas la línea base)

- `dtoverlay=disable-bt` en `/boot/firmware/config.txt` y apagar el power save del Wi-Fi
- Aislar un núcleo para el audio (`isolcpus=3` en `cmdline.txt`) y fijarle el hilo
- Evaluar kernel `PREEMPT_RT` — solo vale la pena si los xruns no se van con lo anterior
- Quitar servicios que no usa un pedal (`triggerhappy`, `avahi` si ya no necesitas `.local`, etc.)

## Etapa 6 — OLED + botones

**Gotcha grande del Pi 5:** el GPIO ya no lo maneja el SoC sino el chip **RP1**.
`RPi.GPIO` y la interfaz sysfs vieja **no funcionan** en Pi 5. Lo vigente es
**libgpiod v2** (`sudo apt install libgpiod-dev gpiod`), y desde Python `lgpio` o
`gpiozero` con backend lgpio.

- OLED SSD1306 por I2C: habilita I2C con `sudo raspi-config` → Interface Options.
  Conecta VCC a 3.3V, GND, SDA al pin 3 (GPIO2), SCL al pin 5 (GPIO3).
  Confirma con `i2cdetect -y 1` → debe aparecer en `0x3C` (a veces `0x3D`).
- Botones: a GND con pull-up interno, y **debounce por software** (ignora cambios
  dentro de ~20 ms).

**Regla arquitectónica que no se rompe:** la UI vive en su propio hilo, nunca dentro
del callback de audio. El callback no puede hacer I2C, ni `printf`, ni reservar
memoria, ni tomar un mutex — cualquiera de esas cosas te cuesta un xrun. Para
cambiar de modelo se carga el nuevo en el hilo de UI y se publica con un
`std::atomic` (exactamente el patrón que usa `ContainerModel` en el core de NAM,
vale la pena leerlo como referencia).

## Etapa 7 — Que arranque solo

Un servicio systemd que lance el binario al bootear, sin login, con reinicio
automático si se cae. Ahí ya es un pedal y no una computadora con una guitarra.

---

## Pendientes de la etapa anterior que siguen abiertos

1. **Noise gate antes del modelo** — el piso de ruido medido fue de −52 dBFS contra
   picos de −16, o sea 36 dB de SNR. Cualquier modelo de alta ganancia lo amplifica.
2. **Normalización de salida** con `GetLoudness()`, para que cambiar de modelo no
   cambie el volumen.
3. **Salida estéreo** — hoy solo se escribe el canal izquierdo.
4. **Etapa de IR** — solo para captures de tipo preamp/amp; los "Full Rig" ya la traen.
