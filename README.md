# NAM Pedal — Prototipo 1: Passthrough

Objetivo de esta etapa: **no procesar nada todavía**. Solo validar que el
pipeline de audio (entrada → callback → salida) corre estable a buffer chico
y medir la latencia real de tu setup actual. Esto es tu línea base — todo lo
que agregues después (NAM, efectos) suma tiempo sobre este número.

## Instalar dependencias

**Windows:** ver la lista de prerequisitos que te dio Claude en el chat
(Visual Studio Build Tools, CMake, Git, vcpkg). Una vez tengas vcpkg:
```powershell
cd C:\dev\vcpkg
.\vcpkg install rtaudio:x64-windows
```

**Linux / Raspberry Pi OS:**
```bash
sudo apt update
sudo apt install librtaudio-dev cmake build-essential pkg-config
```

**macOS:**
```bash
brew install rtaudio cmake pkg-config
```

## Compilar y correr

**Windows (PowerShell, desde la carpeta del proyecto `nam-pedal-proto`):**
```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
.\Release\passthrough.exe
```

Si prefieres usar el botón de VS Code en vez de la terminal: abre la carpeta
`nam-pedal-proto` en VS Code, la extensión CMake Tools te va a preguntar por
un "kit" (elige el de Visual Studio), y en la configuración de la extensión
agrega ese mismo `-DCMAKE_TOOLCHAIN_FILE=...` como argumento de configure.
Luego usa el botón "Build" y "Run" del status bar de abajo.

**Linux / Mac:**
```bash
mkdir build && cd build
cmake ..
make
./passthrough
```

## Qué esperar la primera vez

El programa te va a listar los dispositivos de audio que detecta (busca tu
Scarlett 2i2 en la lista) y va a abrir un stream usando el dispositivo por
defecto de Windows. Si tu 2i2 no es el default del sistema, cámbialo primero
en Configuración de Windows > Sonido, o dime y ajustamos el código para que
elija el dispositivo por índice en vez del default.

Conecta guitarra a la entrada, audífonos (no parlantes, evita feedback) a
la salida.

## Qué hacer con esto

1. Corre a `BUFFER_FRAMES = 128` primero. Debe correr limpio, sin xruns.
2. Baja a `64`, después a `32`. En algún punto tu SO/hardware actual
   (probablemente sin kernel RT) va a empezar a tirar xruns — anota en qué
   número pasa. Ese es tu límite en tu compu de desarrollo, no en el Pi
   (el Pi 5 con kernel RT + interfaz I2S debería aguantar buffers más chicos
   que tu laptop con USB genérico).
3. Anota la latencia algorítmica que imprime el programa y, si puedes,
   mide la latencia real ida-vuelta con un loopback físico (cable de la
   salida a la entrada) y una señal de test — así conoces el offset real
   de driver + hardware sobre el número teórico.

## Siguiente paso (Prototipo 2)

Cuando esto corra estable:
1. Clonar `NeuralAmpModelerCore` (github.com/sdatkinson/NeuralAmpModelerCore)
2. Cargar un archivo `.nam` de prueba (bajá uno chico de tone3000.com)
3. Reemplazar la línea `std::memcpy(out, in, ...)` del callback por la
   llamada al motor de inferencia de NAM
4. Volver a medir xruns al mismo buffer size — la inferencia consume tiempo
   de CPU dentro de la ventana del callback, así que es normal que necesites
   subir el buffer un poco respecto al passthrough puro

Cuando llegues a ese punto, seguimos con la integración de NAM.