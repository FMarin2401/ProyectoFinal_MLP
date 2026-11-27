# V1 · NumPy en CPU (1 hilo)

**Responsable:** Luis Pablo · **Rama:** `numpy`
**Archivos:** `proyecto_mlp_mnist.py` (`backward_numpy`, `sgd_update_numpy`)

## Qué se implementó

### `backward_numpy(p, cache, y)`

Recibe los parámetros `[W1, b1, W2, b2, W3, b3]`, el `cache = (X, Z1, A1, Z2, A2, P)` que devuelve `forward` y las etiquetas `y`. Devuelve `[dW1, db1, dW2, db2, dW3, db3]` en float32.

```python
dZ3 = P.copy()
dZ3[np.arange(B), y] -= 1          # P - one_hot(y), sin construir la matriz one-hot
dZ3 /= B                           # la pérdida es un promedio: el /B va solo aquí
dW3 = A2.T @ dZ3
db3 = dZ3.sum(axis=0)

dZ2 = (dZ3 @ W3.T) * (Z2 > 0)      # dA2 pasando por la ReLU
dW2 = A1.T @ dZ2
db2 = dZ2.sum(axis=0)

dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = dZ1.sum(axis=0)
```

Decisiones:

- **One-hot implícito.** Restar 1 en la posición de la clase correcta equivale a `P − one_hot(y)` y evita crear una matriz (B × 10).
- **`/B` una sola vez.** `forward` devuelve la pérdida ya promediada, así que el factor 1/B entra en `dZ3` y las capas anteriores lo heredan por la regla de la cadena.
- **Sin `dX`.** La entrada no tiene parámetros; calcular `dZ1 @ W1.T` sería una GEMM de (B × 512)·(512 × 784) desperdiciada.
- **float32 de punta a punta.** Todas las operaciones conservan el dtype de los parámetros (`/=` es in-place, la máscara booleana se promueve a float32).

### `sgd_update_numpy(p, grads, lr)`

```python
for w, g in zip(p, grads):
    w -= lr * g
```

Actualiza **in-place**: los arreglos de `p` son los mismos objetos antes y después del paso (no se reasigna la lista), que es lo que espera `NumpyBackend._step`. SGD simple, sin momentum, η = 0.1.

## Verificación

### Gradient check (`python proyecto_mlp_mnist.py --gradcheck`)

Diferencias finitas centradas en float64 (red 20 → 8 → 6 → 4, batch 7, ε = 1e-5, tolerancia 1e-6):

| Parámetro | Error relativo máx. | Estado |
|---|---|---|
| W1 | 4.166e-08 | OK |
| b1 | 7.476e-10 | OK |
| W2 | 1.271e-08 | OK |
| b2 | 6.211e-10 | OK |
| W3 | 1.326e-08 | OK |
| b3 | 3.458e-10 | OK |

**APROBADO** en los 6 parámetros, con dos órdenes de magnitud de margen.

### Entrenamiento corto en MNIST real

`python proyecto_mlp_mnist.py --backend numpy --train-limit 6000 --epochs 3` (batch 128):

| Época | Pérdida |
|---|---|
| 1 | 1.0346 |
| 2 | 0.4511 |
| 3 | 0.3535 |

Accuracy en test: **90.34 %** con solo el 10 % de los datos de entrenamiento. La pérdida baja de forma monótona.

### Un solo hilo

`threadpoolctl` reporta OpenBLAS con `num_threads: 1`: la plantilla fija `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `MKL_NUM_THREADS` antes de importar NumPy.

## FLOPs por iteración

Convención: solo se cuentan las multiplicaciones de matrices; un producto (B × m)·(m × n) cuesta 2·B·m·n FLOPs (una multiplicación y una suma por término).

| GEMM | Capa 1 (784 × 512) | Capa 2 (512 × 256) | Capa 3 (256 × 10) | Subtotal por muestra |
|---|---|---|---|---|
| Forward `Z = A @ W` | 2 · 401,408 | 2 · 131,072 | 2 · 2,560 | 1,070,080 |
| `dW = Aᵀ @ dZ` | 2 · 401,408 | 2 · 131,072 | 2 · 2,560 | 1,070,080 |
| `dA = dZ @ Wᵀ` | — (no se calcula `dX`) | 2 · 131,072 | 2 · 2,560 | 267,264 |
| **Total** | | | | **2,407,424** |

- **Por iteración:** `B · 2,407,424` FLOPs (8 GEMM: 3 del forward, 3 de `dW`, 2 de `dA`).
- **Por época (60,000 muestras):** 60,000 · 2,407,424 ≈ **144.4 GFLOP**, independiente del tamaño de batch.
- Coincide con `flops_per_sample()` de `run_experiments.py` (2,407,424).
- Se ignoran ReLU, softmax, sumas de `db` y SGD: son O(B·n) frente a O(B·m·n) de las GEMM (≈ 1–2 % del total con B = 32, < 1 % desde B = 128; el peso de SGD baja con B porque se hace una vez por iteración).

El backward cuesta ≈ 1.25× el forward (1,337,344 vs 1,070,080 FLOPs por muestra); sin el ahorro de `dX` sería 2×.

## Nota de entorno: pandas 3.x rompe la carga de MNIST

Con **pandas 3.1.1** y **scikit-learn 1.9.1**, `fetch_openml("mnist_784")` falla dentro del parser ARFF de sklearn:

```
TypeError: expected string or bytes-like object, got 'int'
```

Con `pandas<3` (probado con 2.3.3) carga sin problema. `requirements.txt` no fija la versión de pandas, así que una instalación desde un clon limpio puede fallar. Propuesta: fijar `pandas<3` en `requirements.txt`. Con `--synthetic` no se necesita OpenML y no ocurre el error.

## Cómo reproducir

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy==2.4.6 scikit-learn "pandas<3" matplotlib threadpoolctl
python proyecto_mlp_mnist.py --gradcheck
python proyecto_mlp_mnist.py --backend numpy --train-limit 6000 --epochs 3
```
