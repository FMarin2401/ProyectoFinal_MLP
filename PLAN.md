# Plan de trabajo · Proyecto final de Paralelo (MLP en MNIST)

**Equipo:** Francisco Marín · Luis Pablo · Rodrigo Rosales
**Entrega:** Moodle cierra el **martes 1 de diciembre a las 00:00** → entregamos el **lunes 30 en la noche**.
**Enunciado y rúbrica:** [`docs/Proyecto_final.pdf`](docs/Proyecto_final.pdf)

## De qué trata

Entrenar la misma red neuronal (784 → 512 → 256 → 10) sobre MNIST de tres formas: NumPy en CPU con 1 hilo, CuPy en GPU y PyTorch. Luego medir y explicar cuánto acelera la GPU, cuánto cuesta copiar datos CPU ↔ GPU y cómo cambia todo con el tamaño de batch. La plantilla ya trae datos, forward, cronómetro y gráficas; **lo que nos toca programar** son los 5 `# TODO (estudiante)` de `proyecto_mlp_mnist.py`.

## Roles (propuesta, se puede cambiar)

| Quién | Rama | Tareas | Cómo se prueba |
|---|---|---|---|
| **Francisco** | `cupy` | `backward_cupy`, `sgd_update_cupy` · correr **todos** los experimentos en su laptop (RTX 4050) · bonus Numba si hay tiempo | `--consistency` |
| **Luis Pablo** | `numpy` | `backward_numpy`, `sgd_update_numpy` · `--gradcheck` · derivar a mano los FLOPs por iteración para el reporte | `--gradcheck` (no necesita GPU) |
| **Rodrigo** | `pytorch-docs` | `torch_backward_and_update` (2 líneas) · README · reporte (6–10 págs.) y presentación (10 min) | `--consistency` (Colab si no tiene GPU NVIDIA) |

> ‼️ En la defensa le pueden preguntar **a cualquiera** cosas como "deriva dZ3" o "explica la sincronización". Aunque cada quien haga su parte, los tres tenemos que entender el backward.

## Calendario

| Día | Meta | Listo cuando… |
|---|---|---|
| **Jue 26** | Clonar el repo, instalar el entorno, correr la plantilla. Llamada corta para derivar el backward juntos | Todos corren `python proyecto_mlp_mnist.py --backend numpy --synthetic --train-limit 6000 --epochs 1` sin errores |
| **Vie 27** | Llenar los TODO y fusionar las tres ramas en `main` | `--gradcheck` aprobado · `--consistency` < 1e-4 · `run_experiments.py --mode quick` sin errores |
| **Sáb 28** | Correr el Protocolo T y el C en la laptop de Francisco y subir `results/` | CSV y `fig_*.png` en el repo |
| **Dom 29** | Reporte y presentación con los resultados | PDF de 6–10 págs. y diapositivas listas |
| **Lun 30** | Revisión final, congelar el repo, ensayar y **entregar** | Subido a Moodle |

## Cómo trabajamos en el repo

1. Cada quien trabaja en **su rama** (`git checkout -b numpy`, etc.).
2. Cada TODO es una función distinta, así que casi no habrá conflictos. Aun así, **no modifiquen nada fuera de su TODO** sin avisar; el enunciado pide justificar cualquier cambio a la plantilla.
3. Al terminar: `git push` y abrir un *pull request* a `main`. Otro integrante lo revisa y lo fusiona.
4. **No subir** `.venv/`, `.cache/` ni `results_quick/` (ya están en `.gitignore`).
5. Los resultados finales (`results/`) **solo** se generan en la laptop de Francisco, para que la comparación sea válida.

## La parte matemática (para la llamada del jueves)

Convención de la plantilla: `Z = X @ W + b`, cada `W` con forma **(entrada, salida)**, parámetros `[W1, b1, W2, b2, W3, b3]`. El backward de NumPy y el de CuPy es **la misma fórmula** (solo cambia `np` por `cp`), así que lo derivamos una vez:

- Pérdida: softmax + entropía cruzada promediada sobre el batch.
- Salida: `dZ3 = (softmax(Z3) − one_hot(y)) / B`.
- Para cada capa: `dW = A_prevᵀ @ dZ`, `db = suma de dZ sobre el batch`, `dA_prev = dZ @ Wᵀ`.
- ReLU: `dZ = dA * (Z > 0)`.
- SGD: `W ← W − η·dW` con η = 0.1, sin momentum.

Lo que da la plantilla: `forward` devuelve la pérdida **ya promediada** sobre el batch y `cache = (X, Z1, A1, Z2, A2, P)`, con `P = softmax(Z3)`. Por eso el `/ B` va solo en `dZ3` y no se repite en las demás capas. No hace falta calcular `dX` de la primera capa. El orden de salida es `[dW1, db1, dW2, db2, dW3, db3]`, en float32, y SGD actualiza la lista **in-place** (`p[i] -= lr * grads[i]`).

## Checklist de la rúbrica (lo que da puntos)

- [ ] `--gradcheck` aprobado en los 6 parámetros (error relativo < 1e-6, float64)
- [ ] NumPy y CuPy difieren ≤ 0.5 pp en accuracy final · `--consistency` < 1e-4
- [ ] Datos y pesos en VRAM todo el bucle; sin `.item()` por iteración; warm-up y sincronización
- [ ] PyTorch: misma red, pesos copiados (`W.T`), SGD sin momentum, fp32 **sin TF32**
- [ ] Protocolo T: 4 batches (32, 128, 512, 2048), 3 reps GPU, ≥1 NumPy (ideal 2), H→D / D→H / cómputo por separado
- [ ] Protocolo C: batch 128, 20 épocas, 3 backends, curvas de pérdida y accuracy, tiempo hasta 95 %
- [ ] Speedup en cómputo y total por batch; FLOPs derivados a mano; GFLOP/s vs SGEMM y vs pico de la GPU
- [ ] Análisis: subutilización con batch chico, saturación, PCIe, cuBLAS/cuDNN y fusión de kernels
- [ ] README reproducible desde un clon limpio · semillas · versiones · logs y CSV generados por script
- [ ] Reporte 6–10 págs. con ≥ 4 figuras con unidades, hardware completo y **declaración de uso de IA**
- [ ] Presentación de 10 min ± 1 con demo o video
- [ ] (Bonus +10) Kernel Numba `@cuda.jit` de ReLU o gradiente vs CuPy

## Penalizaciones a evitar

Sin semillas (−5) · sin separar transferencia y cómputo (−8) · NumPy multihilo (−8) · no reproducible (−10) · sin curvas o accuracy < 90 % sin explicar (−6) · **IA no declarada (−5)** · −10 % por día de retraso.
