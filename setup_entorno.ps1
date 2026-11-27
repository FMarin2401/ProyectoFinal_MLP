# setup_entorno.ps1 - Crea el entorno del proyecto en Windows y lo verifica.
# Uso (desde la carpeta del repo):
#   powershell -ExecutionPolicy Bypass -File .\setup_entorno.ps1
# Para otra version de CUDA de PyTorch:  -TorchCuda cu130  (ver pytorch.org)
#
# Se puede volver a correr sin miedo: reutiliza .venv, no vuelve a bajar torch si ya
# esta la version con CUDA, y si la descarga de torch (~2.6 GB) se corta, reanuda.

param([string]$TorchCuda = "cu126")

Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

function Paso($titulo, [scriptblock]$accion) {
    Write-Host "`n=== $titulo ===" -ForegroundColor Cyan
    & $accion
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALLO en: $titulo (codigo $LASTEXITCODE). Copia el error de arriba." -ForegroundColor Red
        Write-Host "Si fue la red (download / getaddrinfo / incomplete-download), vuelve a correr el script: retoma donde se quedo." -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $py)) {
    Paso "1/5 Crear .venv" { python -m venv .venv }
} else {
    Write-Host "`n=== 1/5 .venv ya existe, se reutiliza ===" -ForegroundColor Cyan
}
& $py -c "import sys; v = sys.version_info; print('Python', sys.version.split()[0]); v[:2] != (3, 12) and print('AVISO: el proyecto se probo con Python 3.12')"

Paso "2/5 Actualizar pip"   { & $py -m pip install --upgrade pip }
# requirements.txt fija pandas<3 (con pandas 3 no carga MNIST) y numpy 2.4.x: si ya tenias otras
# versiones en el .venv, este paso las corrige.
Paso "3/5 requirements.txt" { & $py -m pip install -r requirements.txt }

# PyTorch con CUDA no esta en PyPI: sale del indice de pytorch.org. Si ya hay un torch de solo CPU,
# pip lo daria por bueno y no bajaria el de CUDA, por eso se desinstala primero.
& $py -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== 4/5 PyTorch con CUDA ya instalado, se omite ===" -ForegroundColor Cyan
} else {
    & $py -m pip uninstall -y -q torch 2>$null
    Paso "4/5 PyTorch ($TorchCuda, ~2.6 GB)" {
        & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$TorchCuda" --resume-retries 20 --timeout 60
    }
}

# Si la maquina no tiene GPU NVIDIA esto solo avisa (NumPy y --gradcheck funcionan igual).
# El aviso "CUDA path could not be detected" de CuPy es inofensivo: [ctk] trae las DLL de CUDA.
Write-Host "`n=== 5/5 Verificar GPU ===" -ForegroundColor Cyan
& $py -W ignore -c @'
import numpy, pandas, torch
print('numpy', numpy.__version__, '| pandas', pandas.__version__)
print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')
try:
    import cupy
    print('cupy', cupy.__version__, '| GPUs:', cupy.cuda.runtime.getDeviceCount())
except Exception as e:
    print('AVISO: CuPy no ve una GPU NVIDIA (' + type(e).__name__ + '). NumPy y --gradcheck si funcionan.')
'@

Write-Host "`n=== Prueba de la plantilla (puede fallar en un TODO que aun no este hecho; eso es normal) ===" -ForegroundColor Cyan
& $py -W ignore proyecto_mlp_mnist.py --backend numpy --synthetic --train-limit 6000 --epochs 1

Write-Host "`nListo. Para usar el entorno:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
