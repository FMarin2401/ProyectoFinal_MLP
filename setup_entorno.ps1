# setup_entorno.ps1 - Crea el entorno del proyecto en Windows y lo verifica.
# Uso (desde la carpeta del repo):
#   powershell -ExecutionPolicy Bypass -File .\setup_entorno.ps1
# Para otra version de CUDA de PyTorch:  -TorchCuda cu130  (ver pytorch.org)

param([string]$TorchCuda = "cu126")

Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

function Paso($titulo, [scriptblock]$accion) {
    Write-Host "`n=== $titulo ===" -ForegroundColor Cyan
    & $accion
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALLO en: $titulo (codigo $LASTEXITCODE). Copia el error de arriba." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $py)) {
    Paso "1/5 Crear .venv" { python -m venv .venv }
} else {
    Write-Host "`n=== 1/5 .venv ya existe, se reutiliza ===" -ForegroundColor Cyan
}

Paso "2/5 Actualizar pip"           { & $py -m pip install --upgrade pip }
Paso "3/5 requirements.txt"         { & $py -m pip install -r requirements.txt }
Paso "4/5 PyTorch ($TorchCuda)"     { & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$TorchCuda" }

Paso "5/5 Verificar GPU" {
    & $py -c 'import torch, cupy, numpy; print("numpy", numpy.__version__); print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NINGUNA"); print("cupy", cupy.__version__, "| GPUs:", cupy.cuda.runtime.getDeviceCount())'
}

Write-Host "`n=== Prueba de la plantilla (puede fallar en un TODO; eso es normal) ===" -ForegroundColor Cyan
& $py proyecto_mlp_mnist.py --backend numpy --synthetic --train-limit 6000 --epochs 1

Write-Host "`nListo. Para usar el entorno:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
