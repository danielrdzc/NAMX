<#
.SYNOPSIS
    Build (y opcionalmente correr) nam-pedal-proto con un solo comando.

.EXAMPLE
    .\build.ps1                 # configura + compila Release
    .\build.ps1 -Run            # compila y arranca el exe
    .\build.ps1 -Clean -Run     # borra build/, recompila desde cero y corre
    .\build.ps1 -Config RelWithDebInfo -Run
    .\build.ps1 -Run -Model models\marshall.nam
#>
[CmdletBinding()]
param(
    [switch]$Run,
    [switch]$Clean,
    [switch]$Bypass,
    [int]$BufferSize = 0,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$Config = 'Release',
    [string]$Model
)

$ErrorActionPreference = 'Stop'
$root     = $PSScriptRoot
$buildDir = Join-Path $root 'build'

function Write-Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }
function Die($msg)        { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# --- prerequisitos ------------------------------------------------------------
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Die "cmake no esta en el PATH. Instalalo o abre una 'Developer PowerShell for VS'."
}

# --- vcpkg --------------------------------------------------------------------
# El CMakeLists ya lo auto-detecta, pero si lo encontramos aca lo exportamos
# para que VS Code / Visual Studio tambien lo vean.
if (-not $env:VCPKG_ROOT) {
    foreach ($candidate in @('C:\dev\vcpkg', "$env:USERPROFILE\vcpkg", 'C:\vcpkg')) {
        if (Test-Path (Join-Path $candidate 'scripts\buildsystems\vcpkg.cmake')) {
            $env:VCPKG_ROOT = $candidate
            break
        }
    }
}
if ($env:VCPKG_ROOT) {
    Write-Host "vcpkg: $env:VCPKG_ROOT" -ForegroundColor DarkGray
} else {
    Write-Host "vcpkg: no encontrado (CMake intentara pkg-config)" -ForegroundColor DarkYellow
}

# --- clean --------------------------------------------------------------------
if ($Clean -and (Test-Path $buildDir)) {
    Write-Step "Borrando $buildDir"
    # vcpkg_installed tarda muchisimo en rearmarse: lo salvamos y lo devolvemos.
    $vcpkgInstalled = Join-Path $buildDir 'vcpkg_installed'
    $stash = $null
    if (Test-Path $vcpkgInstalled) {
        $stash = Join-Path $root '.vcpkg_installed_stash'
        if (Test-Path $stash) { Remove-Item $stash -Recurse -Force }
        Move-Item $vcpkgInstalled $stash
    }
    Remove-Item $buildDir -Recurse -Force
    if ($stash) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
        Move-Item $stash $vcpkgInstalled
    }
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- configure ----------------------------------------------------------------
# Correrlo siempre es barato: si no cambio nada, CMake no hace trabajo.
Write-Step "Configurando ($Config)"
cmake -S $root -B $buildDir -DCMAKE_BUILD_TYPE=$Config
if ($LASTEXITCODE -ne 0) { Die "fallo el configure de CMake." }

# --- build --------------------------------------------------------------------
Write-Step "Compilando"
cmake --build $buildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { Die "fallo la compilacion." }

$sw.Stop()

# --- ubicar el exe ------------------------------------------------------------
$exe = Get-ChildItem -Path $buildDir -Filter 'passthrough.exe' -Recurse -ErrorAction SilentlyContinue |
       Sort-Object LastWriteTime -Descending |
       Select-Object -First 1
if (-not $exe) { Die "compilo pero no encuentro passthrough.exe dentro de build/." }

Write-Ok ("`nBuild OK en {0:N1}s -> {1}" -f $sw.Elapsed.TotalSeconds, $exe.FullName)

# --- run ----------------------------------------------------------------------
if ($Run) {
    Write-Step "Corriendo (Ctrl+C o ENTER para salir)"
    # cwd = raiz del proyecto, asi encuentra test_model.nam / models\
    $exeArgs = @()
    if ($Model)  { $exeArgs += $Model }
    if ($Bypass) { $exeArgs += '--bypass' }
    if ($BufferSize -gt 0) { $exeArgs += @('--buffer', "$BufferSize") }
    Push-Location $root
    try {
        & $exe.FullName @exeArgs
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Para correrlo:  .\build.ps1 -Run" -ForegroundColor DarkGray
}
