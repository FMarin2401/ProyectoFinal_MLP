# Análisis comparativo de cómputo paralelo en el entrenamiento de un MLP sobre MNIST

Proyecto final de **Plataformas y Algoritmos en Paralelo** · Ibero León · Otoño 2026

**Equipo:** Francisco Marín · Luis Pablo · Rodrigo Rosales

Se entrena la misma red (784 → 512 ReLU → 256 ReLU → 10 Softmax) sobre MNIST con tres implementaciones y se comparan tiempos, speedup y convergencia:

| Versión | Backend | Qué se implementa |
|---|---|---|
| V1 | NumPy (CPU, **1 hilo**) | Backward y SGD a mano |
| V2 | CuPy (GPU) | Backward y SGD a mano, datos en VRAM |
| V3 | PyTorch (GPU) | `torch.nn` con SGD sin momentum, fp32 sin TF32 |

El enunciado y la rúbrica están en [`docs/Proyecto_final.pdf`](docs/Proyecto_final.pdf). El plan de trabajo del equipo está en [`PLAN.md`](PLAN.md).

## Estructura

```
proyecto_mlp_mnist.py   # plantilla del profesor + nuestros TODO (backward, SGD, PyTorch)
run_experiments.py      # barrido backends × batch, CSV y gráficas
requirements.txt        # dependencias (con versiones probadas)
setup_entorno.ps1       # crea .venv, instala todo (incluido torch con CUDA) y verifica la GPU
results/                # CSV, logs y figuras del barrido completo (se suben al repo)
docs/                   # enunciado y material del reporte
```

## Instalación (Windows · PowerShell)

Requisitos: Python 3.12 y, para CuPy/PyTorch, una GPU NVIDIA con driver reciente (`nvidia-smi` debe funcionar). **No** hace falta instalar el CUDA Toolkit.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_entorno.ps1
.\.venv\Scripts\Activate.ps1    # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

El script crea `.venv`, instala `requirements.txt`, instala PyTorch con CUDA 12.6 desde pytorch.org (~2.6 GB), verifica la GPU y hace una prueba corta de la plantilla. Se puede correr las veces que haga falta: reutiliza lo que ya está instalado.

Problemas conocidos (ya resueltos en el repo):

| Síntoma | Causa | Qué hacer |
|---|---|---|
| La descarga de torch se corta (`incomplete-download`, `getaddrinfo failed`) | Red inestable; el archivo pesa ~2.6 GB | Volver a correr `setup_entorno.ps1`: reanuda la descarga |
| `TypeError: expected string or bytes-like object, got 'int'` al bajar MNIST | `fetch_openml` no es compatible con pandas 3 | `requirements.txt` ya fija `pandas<3`; correr de nuevo el script (o `pip install -r requirements.txt`) |
| `UserWarning: CUDA path could not be detected` | CuPy no encuentra un CUDA Toolkit instalado | Ignorarlo: `cupy-cuda12x[ctk]` trae sus propias DLL de CUDA |
| La prueba final del script truena con `NotImplementedError` | Algún TODO aún no está en tu rama | Normal hasta que se fusionen las 3 ramas |

Después de cada `git pull` que cambie `requirements.txt` o `setup_entorno.ps1`, vuelve a correr el script.

En Google Colab o Kaggle (plan B sin GPU NVIDIA): `pip install -r requirements.txt` (cupy y torch ya suelen venir instalados).

## Cómo correr

```powershell
# Verificaciones
python proyecto_mlp_mnist.py --gradcheck          # backward de NumPy vs diferencias finitas (float64)
python proyecto_mlp_mnist.py --consistency        # NumPy vs CuPy vs PyTorch tras 3 pasos (< 1e-4)

# Una corrida suelta
python proyecto_mlp_mnist.py --backend cupy --epochs 3 --batch-size 128 --out cupy.json

# Prueba de flujo de todo el pipeline (minutos)
python run_experiments.py --mode quick            # agregar --synthetic si no hay internet/MNIST

# Barrido completo (Protocolos T y C) — solo en la máquina de referencia
python run_experiments.py --mode full --peak-gflops <pico FP32 de la GPU>
python run_experiments.py --mode full --resume    # si se interrumpe
```

Los hilos de CPU se fijan a 1 dentro de la plantilla, antes de importar NumPy (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`). La evidencia queda en `results/env.json`.

## Reproducibilidad

- Semilla 42 en el split, en los pesos He, en el orden de los lotes, en NumPy y en PyTorch (ya fijada en la plantilla).
- Todas las mediciones finales se corren en **la misma máquina y la misma sesión**.
- Hardware de referencia: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, TGP 105 W, driver 610.62) · CPU *(completar)* · Windows 11.
- Software probado: Python 3.12.9 · NumPy 2.4.6 · CuPy 14.2.0 (`cupy-cuda12x[ctk]`) · PyTorch 2.14.1+cu126 · pandas 2.3.3 · scikit-learn 1.9.1. La lista exacta de la corrida final queda en `results/env.json`.

## Créditos y uso de IA

- Plantilla `proyecto_mlp_mnist.py` y `run_experiments.py`: material del profesor del curso.
- Uso de IA: *(declarar aquí y en el reporte: herramienta y para qué se usó)*.
