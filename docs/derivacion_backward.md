# Derivación del backward · MLP 784 → 512 → 256 → 10

Base para `backward_numpy` / `backward_cupy` y para la sección de **método** del reporte. NumPy y CuPy usan exactamente las mismas fórmulas (solo cambia `np` por `cp`); PyTorch las calcula solo con autograd.

## 0. Forward y formas

Convención de la plantilla: `Z = X @ W + b`, con cada `W` de forma **(entrada, salida)**. B = tamaño de batch.

| Paso | Operación | Forma |
|---|---|---|
| Entrada | X | (B, 784) |
| Capa 1 | Z1 = X·W1 + b1 · A1 = ReLU(Z1) | (B, 512) |
| Capa 2 | Z2 = A1·W2 + b2 · A2 = ReLU(Z2) | (B, 256) |
| Capa 3 | Z3 = A2·W3 + b3 · P = softmax(Z3) | (B, 10) |
| Pérdida | L = −(1/B) Σᵢ log P[i, yᵢ] | escalar |

`forward` devuelve L **ya promediada** y `cache = (X, Z1, A1, Z2, A2, P)`.

## 1. Salida: softmax + entropía cruzada

$$dZ_3 = \frac{\partial L}{\partial Z_3} = \frac{P - Y}{B}$$

donde Y es el one-hot de las etiquetas.

- **Intuición:** "lo que predije menos lo que debía predecir". Si a la clase correcta le dio 0.7, el gradiente ahí es −0.3 (súbela); si a una incorrecta le dio 0.2, es +0.2 (bájala).
- **Demostración corta (una muestra):** con ℓ = −log p_y y p_c = e^{z_c} / Σ_k e^{z_k},
  ∂ℓ/∂z_c = p_c − 1[c = y]. Promediar sobre el batch agrega el factor 1/B.
- **El `/B` va solo aquí.** Todo lo demás se deriva de dZ3, así que ya lo hereda. Repetirlo es el error más común.
- **En código** no hace falta construir Y: se copia P, se resta 1 en las posiciones `[arange(B), y]` y se divide entre B.

## 2. Capa lineal (la regla de todas las capas)

Para Z = A_prev·W + b:

| Gradiente | Fórmula | Forma en la capa 3 |
|---|---|---|
| dW | A_prevᵀ · dZ | (256, B)·(B, 10) = (256, 10) = forma de W3 ✓ |
| db | Σ_batch dZ (suma sobre el eje 0) | (10,) = forma de b3 ✓ |
| dA_prev | dZ · Wᵀ | (B, 10)·(10, 256) = (B, 256) = forma de A2 ✓ |

- **Chequeo de formas:** cada gradiente tiene la forma de su variable, y las dimensiones solo encajan de una manera. Si dudas del orden de un producto, revisa las formas.
- **¿Por qué db es una suma?** b se suma (broadcasting) a las B filas del batch: se usó B veces, así que su gradiente acumula las B contribuciones.

## 3. ReLU: la compuerta

$$dZ = dA \odot \mathbb{1}[Z > 0]$$

Si la neurona estaba apagada (Z ≤ 0) no influyó en la salida y no pasa gradiente; si estaba prendida lo deja pasar tal cual (la derivada de x es 1). Por eso `cache` guarda Z1 y Z2.

## 4. Hacia atrás, capa por capa

| Capa | Gradientes |
|---|---|
| 3 | dW3 = A2ᵀ·dZ3 · db3 = Σ dZ3 · dA2 = dZ3·W3ᵀ |
| ReLU 2 | dZ2 = dA2 ⊙ (Z2 > 0) |
| 2 | dW2 = A1ᵀ·dZ2 · db2 = Σ dZ2 · dA1 = dZ2·W2ᵀ |
| ReLU 1 | dZ1 = dA1 ⊙ (Z1 > 0) |
| 1 | dW1 = Xᵀ·dZ1 · db1 = Σ dZ1 · **sin dX** (X no se entrena) |

Salida: `[dW1, db1, dW2, db2, dW3, db3]`, en ese orden y en float32.

## 5. SGD

$$\theta \leftarrow \theta - \eta\,\nabla_\theta L, \quad \eta = 0.1,\ \text{sin momentum}$$

Se actualiza **in-place** (`p[i] -= lr * grads[i]`): la plantilla sigue usando la misma lista `p`, y en CuPy así no se reserva memoria nueva en cada iteración.

## 6. Todo el backward en una lista (para escribir sin ver)

1. dZ3 = (P − Y) / B
2. dW3 = A2ᵀ · dZ3  ·  db3 = Σ dZ3  ·  dA2 = dZ3 · W3ᵀ
3. dZ2 = dA2 ⊙ (Z2 > 0)
4. dW2 = A1ᵀ · dZ2  ·  db2 = Σ dZ2  ·  dA1 = dZ2 · W2ᵀ
5. dZ1 = dA1 ⊙ (Z1 > 0)
6. dW1 = Xᵀ · dZ1  ·  db1 = Σ dZ1
7. SGD: p[i] −= η · grads[i]

En realidad son solo 3 reglas (salida, capa lineal y ReLU) repetidas hacia atrás.

## 7. Costo en FLOPs (para el reporte)

Un producto (B×m)·(m×n) cuesta 2·B·m·n FLOPs. Por iteración hay 3 GEMM en el forward, 3 de dW y solo 2 de dA (la capa 1 no calcula dX):

| | Capa 1 (784×512) | Capa 2 (512×256) | Capa 3 (256×10) |
|---|---|---|---|
| Forward | 2B·401,408 | 2B·131,072 | 2B·2,560 |
| dW | 2B·401,408 | 2B·131,072 | 2B·2,560 |
| dA | — | 2B·131,072 | 2B·2,560 |

Total ≈ **B · 2,407,424 FLOPs por iteración**. Con 60,000 imágenes por época: ≈ **144 GFLOP por época** (independiente de B). Se ignoran las operaciones elemento a elemento (ReLU, softmax, SGD), que son de orden B·n y pesan poco frente a los GEMM.

## 8. Errores típicos (lo que delata `--gradcheck`)

| Síntoma | Causa probable |
|---|---|
| Error de formas en un `@` | transpuesta en el lado equivocado (`dZ @ A_prev` en vez de `A_prev.T @ dZ`) |
| Todos los gradientes salen B veces más chicos | `/B` repetido en cada capa |
| Fallan solo los `db` | sumar sobre el eje 1 en vez del 0, o usar `mean` |
| Fallan dW1/db1 pero dW3 está bien | máscara ReLU con A en vez de Z, o con la Z de otra capa |
| CuPy lento o `--consistency` raro | `.get()` / `.item()` / NumPy mezclado dentro del backward |
