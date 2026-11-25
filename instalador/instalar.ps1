# Instalador de Charter
# Todo (Python, librerias, ffmpeg, modelos y caches) se instala dentro de la carpeta del programa.
# Para desinstalar basta con borrar la carpeta.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root "runtime"
$Models = Join-Path $Root "models"
$UvDir = Join-Path $Runtime "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$Venv = Join-Path $Runtime "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$FfmpegDir = Join-Path $Runtime "ffmpeg"

# Caches y descargas dentro de la carpeta
$env:UV_CACHE_DIR = Join-Path $Runtime "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
$env:UV_PYTHON_BIN_DIR = Join-Path $Runtime "python-bin"
$env:UV_TOOL_DIR = Join-Path $Runtime "uv-tools"
$env:UV_LINK_MODE = "copy"
$env:HF_HOME = Join-Path $Models "huggingface"
$env:TORCH_HOME = Join-Path $Models "torch"
$env:PIP_CACHE_DIR = Join-Path $Runtime "pip-cache"
$env:CHARTER_ROOT = $Root

function Paso($texto) { Write-Host ""; Write-Host "==> $texto" -ForegroundColor Cyan }
function Ok($texto) { Write-Host "    $texto" -ForegroundColor Green }
function Aviso($texto) { Write-Host "    $texto" -ForegroundColor Yellow }

function Descargar($url, $destino) {
    Write-Host "    Descargando / Downloading $url"
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $destino -UseBasicParsing
            return
        } catch {
            if ($i -eq 3) { throw }
            Aviso "Fallo la descarga (intento $i), reintentando... / Download failed (try $i), retrying..."
            Start-Sleep -Seconds 3
        }
    }
}

Write-Host "==============================================" -ForegroundColor Magenta
Write-Host "   Instalador de Charter / Charter installer  " -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta
Write-Host "Carpeta / Folder: $Root"

New-Item -ItemType Directory -Force $Runtime, $Models, $UvDir | Out-Null

# ---------------------------------------------------------------- GPU / CUDA
Paso "Comprobando tarjeta grafica NVIDIA y driver / Checking NVIDIA graphics card and driver"
$gpuOk = $false
try {
    $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $smi) {
        $partes = ($smi | Select-Object -First 1).Split(",") | ForEach-Object { $_.Trim() }
        $nombre = $partes[0]; $driver = $partes[1]; $vram = $partes[2]
        Ok "GPU: $nombre ($vram), driver $driver"
        $mayor = [int]($driver.Split(".")[0])
        if ($mayor -lt 570) {
            Aviso "Tu driver de NVIDIA es antiguo ($driver). Charter necesita la version 570 o superior (CUDA 12.8)."
            Aviso "Your NVIDIA driver is old ($driver). Charter needs version 570 or newer (CUDA 12.8)."
            Aviso "Se abrira la pagina de descarga de drivers de NVIDIA. Instala el driver y vuelve a ejecutar el instalador."
            Aviso "The NVIDIA driver download page will open. Install the driver and run the installer again."
            Start-Process "https://www.nvidia.com/es-es/drivers/"
            Read-Host "Pulsa Enter para continuar de todos modos (funcionara solo con CPU, muy lento)"
        } else {
            $gpuOk = $true
        }
    }
} catch { }
if (-not $gpuOk -and -not $smi) {
    Aviso "No se ha encontrado una tarjeta NVIDIA con driver instalado. / No NVIDIA card with a driver was found."
    Aviso "Charter funcionara con CPU, pero sera MUCHO mas lento (puede tardar muchas horas por cancion)."
    Aviso "Charter will run on the CPU, but MUCH slower (it can take many hours per song)."
    Aviso "Si tienes una NVIDIA, instala su driver / If you have an NVIDIA card, install its driver: https://www.nvidia.com/drivers/"
}
Ok "No hace falta instalar CUDA / No need to install CUDA: PyTorch brings its own."

# ---------------------------------------------------------------- uv
Paso "Preparando gestor de paquetes / Preparing package manager (uv)"
if (-not (Test-Path $UvExe)) {
    $zip = Join-Path $Runtime "uv.zip"
    Descargar "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" $zip
    Expand-Archive -Path $zip -DestinationPath $UvDir -Force
    Remove-Item $zip
}
Ok (& $UvExe --version)

# ---------------------------------------------------------------- Python
Paso "Instalando / Installing Python 3.12 portable"
if (-not (Test-Path $Py)) {
    & $UvExe python install 3.12 --no-bin
    if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar Python" }
    & $UvExe venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno de Python" }
}
Ok (& $Py --version)

# ---------------------------------------------------------------- ffmpeg
Paso "Instalando / Installing ffmpeg portable"
if (-not (Test-Path (Join-Path $FfmpegDir "ffmpeg.exe"))) {
    $zip = Join-Path $Runtime "ffmpeg.zip"
    Descargar "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" $zip
    $tmp = Join-Path $Runtime "ffmpeg-tmp"
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force $FfmpegDir | Out-Null
    Get-ChildItem -Path $tmp -Recurse -Include ffmpeg.exe, ffprobe.exe | Copy-Item -Destination $FfmpegDir -Force
    Remove-Item $tmp -Recurse -Force
    Remove-Item $zip
}
Ok "ffmpeg listo / ready"
$env:PATH = "$FfmpegDir;$env:PATH"

# ---------------------------------------------------------------- Librerias
Paso "Instalando librerias / Installing libraries (PyTorch + CUDA, separation, transcription, interface...)"
Aviso "Esto puede tardar bastante la primera vez (varios GB). / This can take a while the first time (several GB)."
$backend = if ($gpuOk) { "cu128" } else { "cpu" }
$req = Join-Path $Root "app\requirements.txt"
& $UvExe pip install --python $Py --torch-backend=$backend -r $req
if ($LASTEXITCODE -ne 0) { throw "Fallo la instalacion de librerias / Library installation failed" }
Ok "Librerias instaladas / Libraries installed"

Paso "Comprobando PyTorch y la GPU / Checking PyTorch and the GPU"
& $Py -c "import torch;print('    PyTorch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------------------------------------------- Modelos
Paso "Descargando modelos de IA / Downloading AI models (several GB)"
$env:PYTHONPATH = Join-Path $Root "app"
& $Py -m charter.setup_models
if ($LASTEXITCODE -ne 0) {
    Aviso "Algunos modelos no se pudieron descargar. Se reintentara automaticamente al usar el programa."
    Aviso "Some models couldn't be downloaded. It will retry automatically when you use the program."
}

# ---------------------------------------------------------------- Modelos de Charter
Paso "Descargando los modelos de Charter / Downloading Charter's trained models"
$Config = Join-Path $Root "config"
if (-not (Test-Path (Join-Path $Config "notas_neuronal.pt"))) {
    New-Item -ItemType Directory -Force $Config | Out-Null
    $zip = Join-Path $Runtime "charter-modelos.zip"
    Descargar "https://github.com/Biithh/Auto-Charter/releases/latest/download/charter-modelos.zip" $zip
    Expand-Archive -Path $zip -DestinationPath $Config -Force
    Remove-Item $zip
}
Ok "Modelos de Charter listos / Charter models ready"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host " Instalacion terminada / Installation finished " -ForegroundColor Green
Write-Host " Abre / Open 'Charter.bat'                    " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
