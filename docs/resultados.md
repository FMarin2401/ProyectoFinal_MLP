# Resultados del barrido completo

Resumen de lo que hay en [`results/`](../results/) para armar el reporte. Todos los números salen de los CSV generados por `run_experiments.py`. Si se vuelve a correr el barrido, hay que actualizar esta página.

## Cómo se generaron

```powershell
# Protocolos T y C (batch 128) + referencia SGEMM
python run_experiments.py --mode full --peak-gflops 12100 --reps-numpy 2
# Protocolo C en GPU con los 4 batches (reutiliza lo anterior)
python run_experiments.py --mode full --peak-gflops 12100 --reps-numpy 2 --conv-batches 32 128 512 2048 --skip-gemm --skip-timing --resume
```

Corrido el **jueves 26 de noviembre de 2026**, en la misma sesión y la misma máquina, con la laptop conectada a la corriente. Duró unos 9 minutos en total.

## Hardware y software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU · 2560 núcleos CUDA · 6 GB GDDR6 · TGP 105 W · PCIe 4.0 x8 |
| Driver / CUDA | Driver 610.62 · CUDA runtime de CuPy 12.9 · PyTorch con CUDA 12.6 y cuDNN 9 |
| CPU | Intel Core i5-13450HX (10 núcleos / 16 hilos); **NumPy fijado a 1 hilo** (ver `results/env.json`: OpenBLAS y OpenMP con `num_threads: 1`) |
| RAM / equipo | 12 GB · laptop Lenovo 83DV · Windows 11 (build 26200) |
| Software | Python 3.12.9 · NumPy 2.4.6 · CuPy 14.2.0 · PyTorch 2.14.1+cu126 · scikit-learn 1.9.1 · pandas 2.3.3 |

**Pico FP32 teórico de la GPU:** 2 FLOP (FMA) × 2560 núcleos × 2.37 GHz (boost de la hoja de datos de NVIDIA) ≈ **12,100 GFLOP/s**. `nvidia-smi` reporta un reloj máximo de 3105 MHz (≈ 15,900 GFLOP/s), pero ese reloj no se sostiene en una laptop, así que usamos el boost oficial.

**Referencia SGEMM medida (cuBLAS vía CuPy, 4096³):** **6,747 GFLOP/s**, el 56 % del pico teórico. SGEMM con las formas de nuestra red ([`hardware_peak.json`](../results/hardware_peak.json)): 1,531 (B=32) · 2,923 (B=128) · 5,237 (B=512) · 6,880 (B=2048) GFLOP/s.

**FLOPs** (derivados por Luis Pablo en [`numpy.md`](numpy.md)): `B · 2,407,424` por iteración → **144.4 GFLOP por época** (60,000 muestras), independiente del batch.

## Figuras

| Archivo | Qué muestra | Úsala para |
|---|---|---|
| [`fig_speedup.png`](../results/fig_speedup.png) | Speedup contra NumPy, en cómputo y en total, por batch | Resultado principal |
| [`fig_desglose.png`](../results/fig_desglose.png) | Cómputo contra transferencia H↔D en GPU, y tiempo total por backend | Costo de PCIe |
| [`fig_gflops.png`](../results/fig_gflops.png) | GFLOP/s alcanzados contra SGEMM y contra el pico | Eficiencia y saturación |
| [`fig_convergencia.png`](../results/fig_convergencia.png) | Pérdida y accuracy por época (batch 128, 3 backends) y accuracy contra tiempo | Correctitud y convergencia |
| [`fig_convergencia_batches.png`](../results/fig_convergencia_batches.png) | Accuracy por época y contra tiempo, para los 4 batches en GPU | Equilibrio entre batch y convergencia |

## Protocolo T: tiempos y speedup

3 épocas de MNIST completo (60,000 muestras), 3 repeticiones en GPU y 2 en NumPy. Media ± desviación estándar del cómputo. Fuente: [`summary.csv`](../results/summary.csv).

| Backend | Batch | Cómputo (s) | Total (s) | H→D (s) | GFLOP/s | Speedup cómputo | Speedup total | % de SGEMM | % del pico |
|---|---|---|---|---|---|---|---|---|---|
| NumPy | 32 | 12.38 ± 2.05 | 12.38 | — | 35 | 1× | 1× | 0.5 | 0.3 |
| NumPy | 128 | 6.12 ± 0.46 | 6.12 | — | 71 | 1× | 1× | 1.1 | 0.6 |
| NumPy | 512 | 5.37 ± 0.44 | 5.37 | — | 81 | 1× | 1× | 1.2 | 0.7 |
| NumPy | 2048 | 4.90 ± 0.07 | 4.90 | — | 88 | 1× | 1× | 1.3 | 0.7 |
| CuPy | 32 | 8.10 ± 0.74 | 8.18 | 0.081 | 54 | 1.5× | 1.5× | 0.8 | 0.4 |
| CuPy | 128 | 2.08 ± 0.16 | 2.17 | 0.092 | 210 | 2.9× | 2.8× | 3.1 | 1.7 |
| CuPy | 512 | 0.49 ± 0.04 | 0.57 | 0.075 | 888 | 11.0× | 9.5× | 13.2 | 7.3 |
| CuPy | 2048 | 0.17 ± 0.03 | 0.25 | 0.080 | 2,648 | 29.2× | 19.7× | 39.3 | 21.9 |
| PyTorch | 32 | 6.11 ± 0.14 | 6.16 | 0.042 | 71 | 2.0× | 2.0× | 1.1 | 0.6 |
| PyTorch | 128 | 1.54 ± 0.03 | 1.58 | 0.042 | 282 | 4.0× | 3.9× | 4.2 | 2.3 |
| PyTorch | 512 | 0.41 ± 0.00 | 0.45 | 0.042 | 1,065 | 13.2× | 11.9× | 15.8 | 8.8 |
| PyTorch | 2048 | 0.12 ± 0.01 | 0.16 | 0.043 | **3,669** | **41.4×** | **30.1×** | 54.4 | 30.3 |

D→H (bajar los pesos al final) cuesta ≈ 1–2 ms en todos los casos: es despreciable.

## Protocolo C: convergencia (20 épocas)

Fuente: [`convergence.csv`](../results/convergence.csv) y [`time_to_acc.csv`](../results/time_to_acc.csv).

| Backend | Batch | Época en que llega a 95 % | Cómputo hasta 95 % (s) | Accuracy final | Cómputo de 20 épocas (s) |
|---|---|---|---|---|---|
| NumPy | 128 | 2 | 3.82 | 98.09 % | 40.2 |
| CuPy | 32 | 1 | 2.78 | 98.34 % | 65.6 |
| CuPy | 128 | 2 | 1.33 | 98.11 % | 12.4 |
| CuPy | 512 | 9 | 1.78 | 96.91 % | 3.7 |
| CuPy | 2048 | no llega | — | 94.49 % | 1.0 |
| PyTorch | 32 | 1 | 1.93 | 98.40 % | 41.8 |
| PyTorch | 128 | 2 | **1.06** | 98.07 % | 9.7 |
| PyTorch | 512 | 9 | 1.25 | 96.91 % | 2.7 |
| PyTorch | 2048 | no llega | — | 94.48 % | 0.7 |

- Con batch 128, las curvas de pérdida de los 3 backends se enciman y la accuracy final difiere ≤ 0.04 pp (NumPy 98.09 %, CuPy 98.11 %, PyTorch 98.07 %). La rúbrica pide ≤ 0.5 pp entre NumPy y CuPy. Esto coincide con `--consistency`, que da 1.5e-08 contra una tolerancia de 1e-4.
- **Batch 2048 no llega a 95 % en 20 épocas** (94.5 %). No es un error: con lr = 0.1 fijo solo hace 30 actualizaciones por época, 600 en total, contra 9,380 de batch 128. Está por encima del 90 % que exige la rúbrica, pero **hay que explicarlo en el reporte**.

## Puntos para el análisis (con cálculos)

La rúbrica pide discutir: subutilización con batch chico, saturación, PCIe, cuBLAS/cuDNN y fusión de kernels.

### 1. Con batch chico la GPU está casi vacía: domina el overhead, no el cómputo

Tiempo de cómputo **por iteración** (cómputo ÷ iteraciones; 3 épocas = 5,625 iteraciones con B=32 y 90 con B=2048):

| Batch | NumPy (ms/iter) | CuPy (ms/iter) | PyTorch (ms/iter) |
|---|---|---|---|
| 32 | 2.2 | 1.44 | 1.09 |
| 128 | 4.3 | 1.48 | 1.09 |
| 512 | 15.2 | 1.39 | 1.15 |
| 2048 | 54.5 | 1.86 | 1.31 |

- En **GPU el tiempo por iteración es casi constante** (≈ 1.1–1.9 ms) aunque el trabajo crece **64×** de B=32 a B=2048. El costo lo pone el número de iteraciones (lanzar kernels, overhead de Python y de CuPy/PyTorch), no la aritmética. Con B=32 cada GEMM es tan chica que no llena los SMs de la GPU, por eso solo se llega al 0.4 % del pico.
- En **NumPy el tiempo crece con el batch** (2.2 → 54.5 ms) porque la CPU sí está limitada por el cómputo. Por eso el speedup sube con el batch: de 1.5–2× a 29–41×.

### 2. Saturación

- Los GFLOP/s de la GPU suben casi linealmente con el batch (54 → 2,648 en CuPy) y todavía no se aplanan en B=2048. La red no satura la GPU ni con el batch más grande: el mejor caso (PyTorch, B=2048) llega al 54 % de la SGEMM medida y al 30 % del pico teórico.
- La propia SGEMM con las formas de la red también crece con el batch (1,531 → 6,880 GFLOP/s): las GEMM de (B × 784)·(784 × 512) con B chico no tienen suficiente paralelismo.
- NumPy en 1 hilo se satura pronto: 71 → 88 GFLOP/s de B=128 a B=2048.

### 3. Transferencias por PCIe

- Datos que se suben una vez: X de train (60,000 × 784 float32 = 188 MB) + X de test (31 MB) + etiquetas ≈ **220 MB**. H→D tarda ≈ 0.08 s en CuPy (**≈ 2.7 GB/s**) y ≈ 0.042 s en PyTorch (**≈ 5.2 GB/s**). El enlace es PCIe 4.0 x8, ≈ 15.75 GB/s teóricos. Nos quedamos en 17–33 % porque se copia desde memoria normal del sistema (pageable) y no desde memoria fijada (pinned).
- La transferencia es fija (no depende del batch), pero el cómputo baja con el batch. Por eso pesa cada vez más: en CuPy con B=32 es el 1 % del total, con B=2048 es el **32 %** (0.080 de 0.249 s). Eso explica por qué el speedup total (19.7×) es mucho menor que el de cómputo (29.2×) con batch grande.
- Gracias a que los datos y pesos se quedan en VRAM todo el entrenamiento (sin `.get()` ni `.item()` por iteración), la transferencia se paga **una sola vez** y no en cada lote.

### 4. CuPy contra PyTorch: cuBLAS y fusión de kernels

- Los dos usan cuBLAS para las GEMM, pero **PyTorch es más rápido en todos los batches**: 1.2–1.4× en cómputo.
- En CuPy, cada operación elemental del backward (`P.copy()`, restar el one-hot, `/B`, la máscara `Z > 0`, la multiplicación, las sumas de `db`, el `-=` del SGD) es **un kernel distinto** con su propia lectura y escritura de memoria. Además, cada una pasa por el intérprete de Python.
- PyTorch usa kernels más especializados: `nn.CrossEntropyLoss` combina log-softmax y NLL, y `nn.Linear` suma el bias dentro de la GEMM. Su autograd corre en C++, y el SGD actualiza todos los parámetros con kernels *foreach* (varios tensores por kernel). Son menos kernels y menos overhead por iteración: se ve en que su ms/iter es menor en todos los batches.
- cuDNN no interviene: la red no tiene convoluciones, todo son GEMM de cuBLAS y kernels elementales.
- También: el H→D de PyTorch es ≈ 2× más rápido que el de CuPy con los mismos datos.

### 5. Equilibrio entre batch y convergencia (qué batch conviene)

- El batch más rápido por época (2048) **no es el que llega antes a una buena accuracy**. En 20 épocas no pasa de 94.5 % porque hace 16× menos actualizaciones de pesos que B=128 con el mismo learning rate.
- B=32 converge en menos épocas (95 % en la primera), pero cada época es lenta porque la GPU está subutilizada.
- **B=128 es el mejor equilibrio**: llega a 95 % en ≈ 1.1 s (PyTorch) o 1.3 s (CuPy), contra 3.8 s de NumPy.

### 6. Variabilidad

- PyTorch es muy estable (desviación ≤ 5 % de la media). NumPy con batch chico y CuPy con batch grande varían más (NumPy B=32: ± 17 %; CuPy B=2048: ± 20 %, porque el tiempo es tan corto, 0.17 s, que cualquier ruido del sistema pesa).
- Se usaron 3 repeticiones en GPU y 2 en NumPy, con warm-up y `synchronize` antes de cada medición (lo hace la plantilla).

### 7. Limitaciones del hardware y del experimento

- **GPU de laptop:** el reloj real depende de la temperatura y del límite de potencia (105 W), así que el pico de 12,100 GFLOP/s es optimista. Por eso también comparamos contra la SGEMM *medida* (6,747 GFLOP/s), que es una referencia más justa.
- **Windows (WDDM):** en Windows el driver agrupa los lanzamientos de kernels, lo que agrega latencia por kernel. Esto empeora el caso de batch chico, donde hay muchos kernels pequeños. No lo medimos por separado: es una hipótesis para explicar el ms/iter casi constante de la GPU.
- **Memoria no fijada (pageable):** las copias H→D usan memoria normal del sistema, así que no alcanzan el ancho de banda de PCIe 4.0 x8 (≈ 2.7–5.2 de ≈ 15.75 GB/s).
- **VRAM de 6 GB:** alcanza de sobra (MNIST en float32 ≈ 220 MB), pero limitaría con datasets más grandes.
- **El CPU tiene 16 hilos, pero NumPy usa 1** (lo pide el enunciado). El speedup es contra un solo núcleo, no contra el CPU completo.
- **Pocas repeticiones:** 3 en GPU y 2 en NumPy; el Protocolo C se corrió una vez por configuración.

## Esquema de épocas (justificación pedida por la rúbrica)

El Protocolo T usa 3 épocas porque mide **velocidad**: los FLOPs por época son constantes (≈ 144.4 GFLOP) sin importar el batch, así que más épocas no cambian el speedup. La **precisión** se mide en el Protocolo C (20 épocas). Con η fijo, un batch grande hace menos actualizaciones por época (1,875 / 469 / 118 / 30 para B = 32 / 128 / 512 / 2048), por eso su accuracy en el Protocolo T es menor (97.4 % → 89.2 %). El enunciado dice que esto es un hallazgo y no un error.
