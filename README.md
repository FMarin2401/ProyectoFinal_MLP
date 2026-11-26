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
results/                # CSV, logs y figuras del barrido completo (se suben al repo)
docs/                   # enunciado y material del reporte
```

## Instalación (Windows · PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate          # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
# PyTorch con CUDA: copiar el comando de https://pytorch.org/get-started/locally/ (Windows, Pip, CUDA 12.x)
nvidia-smi                      # anotar modelo de GPU, driver y versión de CUDA para el reporte
```

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
- Hardware de referencia: *(completar: GPU, CPU, driver, CUDA, versiones de librerías)*.

## Créditos y uso de IA

- Plantilla `proyecto_mlp_mnist.py` y `run_experiments.py`: material del profesor del curso.
- Uso de IA: *(declarar aquí y en el reporte: herramienta y para qué se usó)*.
