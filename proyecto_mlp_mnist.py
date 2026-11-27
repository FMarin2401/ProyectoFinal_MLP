#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto Final de Cómputo Paralelo
Análisis comparativo de cómputo paralelo en el entrenamiento de un MLP sobre MNIST

PLANTILLA PARA EL ESTUDIANTE
----------------------------
Arquitectura : 784 -> 512 (ReLU) -> 256 (ReLU) -> 10 (Softmax + entropía cruzada)
Convención   : cada W tiene forma (entrada, salida);  Z = X @ W + b
Parámetros   : lista plana [W1, b1, W2, b2, W3, b3]
Precisión    : float32 en TODO el entrenamiento (la verificación de gradientes usa
               float64 solo para que las diferencias finitas sean confiables).

Lo único que debes escribir son los bloques marcados "# TODO (estudiante)":
  1) backward_numpy  y  sgd_update_numpy
  2) backward_cupy   y  sgd_update_cupy
  3) torch_backward_and_update   (dos líneas)
El resto (datos, pesos iniciales, orden de los lotes, cronometraje, evaluación)
ya está resuelto para que las tres versiones sean comparables. NO lo modifiques
sin justificarlo en el reporte.

Uso rápido
----------
  python proyecto_mlp_mnist.py --gradcheck                       # tras escribir backward_numpy
  python proyecto_mlp_mnist.py --backend numpy --epochs 1 --batch-size 128
  python proyecto_mlp_mnist.py --backend cupy  --epochs 3 --batch-size 128 --out cupy.json
  python proyecto_mlp_mnist.py --consistency                     # compara los backends tras 3 pasos
  python proyecto_mlp_mnist.py --backend numpy --synthetic --train-limit 6000 --epochs 1
      (--synthetic genera datos tipo MNIST sin internet: SOLO para probar el flujo)

El script devuelve (impreso y, con --out, en JSON) un diccionario con:
  transfer_s, compute_s, total_s, test_acc, loss_por_época  (+ detalles extra)
"""

# --------------------------------------------------------------------------
# 1) Hilos de CPU: DEBE ir ANTES de importar NumPy (o cualquier librería BLAS)
# --------------------------------------------------------------------------
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 2) Configuración congelada (no cambiar entre versiones)
# --------------------------------------------------------------------------
SEED = 42
ARCH = (784, 512, 256, 10)
LR = 0.1                      # tasa de aprendizaje constante (SGD manual, sin momentum)
N_TRAIN, N_TEST = 60_000, 10_000
CACHE_DIR = Path(".cache")
WARMUP_ITERS = 5


def thread_report():
    """Evidencia de que el baseline es de un solo hilo (guárdala en el reporte)."""
    info = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [
            {"api": p.get("internal_api"), "num_threads": p.get("num_threads")}
            for p in threadpool_info()
        ]
    except Exception:
        info["threadpools"] = "threadpoolctl no instalado (pip install threadpoolctl)"
    return info


# --------------------------------------------------------------------------
# 3) Datos
# --------------------------------------------------------------------------
def _synthetic_mnist_like(seed=SEED, n=N_TRAIN + N_TEST):
    """Datos sintéticos tipo MNIST (784 features en [0,1], 10 clases). Solo para pruebas de flujo."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(10, 784))
    y = np.arange(n) % 10
    X = centers[y] + rng.normal(0.0, 2.0, size=(n, 784))
    X = (1.0 / (1.0 + np.exp(-X))).astype(np.float32)
    return X, y.astype(np.int64)


def load_mnist(synthetic=False, train_limit=None, test_limit=None):
    """Devuelve Xtr, ytr, Xte, yte (float32 en [0,1], int64). Split estratificado 60k/10k, semilla 42.

    El split se guarda en .cache/ para que las tres versiones (y todas las corridas)
    usen EXACTAMENTE los mismos índices.
    """
    from sklearn.model_selection import train_test_split

    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / ("split_synthetic.npz" if synthetic else "mnist_784_split.npz")
    if cache.exists():
        z = np.load(cache)
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    else:
        if synthetic:
            X, y = _synthetic_mnist_like()
        else:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
            X = (np.asarray(X, dtype=np.float32) / 255.0).astype(np.float32)
            y = np.asarray(y).astype(np.int64)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=N_TRAIN, test_size=N_TEST, stratify=y, random_state=SEED
        )
        Xtr, Xte = np.ascontiguousarray(Xtr, np.float32), np.ascontiguousarray(Xte, np.float32)
        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    if train_limit:
        Xtr, ytr = Xtr[:train_limit], ytr[:train_limit]
    if test_limit:
        Xte, yte = Xte[:test_limit], yte[:test_limit]
    assert Xtr.dtype == np.float32 and Xtr.min() >= 0.0 and Xtr.max() <= 1.0
    return Xtr, ytr, Xte, yte


# --------------------------------------------------------------------------
# 4) Pesos iniciales compartidos (He, semilla fija): se generan UNA vez en CPU
# --------------------------------------------------------------------------
def he_init(arch=ARCH, seed=SEED, dtype=np.float32):
    """[W1, b1, W2, b2, W3, b3] con W de forma (entrada, salida) y b = 0."""
    rng = np.random.default_rng(seed)
    params = []
    for fan_in, fan_out in zip(arch[:-1], arch[1:]):
        W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        params += [W.astype(dtype), np.zeros(fan_out, dtype=dtype)]
    return params


def epoch_permutation(epoch, n):
    """Mismo orden de lotes en las tres versiones (se genera en CPU y se sube a la GPU)."""
    return np.random.default_rng(SEED + epoch).permutation(n)


# --------------------------------------------------------------------------
# 5) Forward + pérdida (dado). Sirve con xp = numpy o cupy.
# --------------------------------------------------------------------------
def forward(xp, p, X, y):
    """Devuelve (loss_escalar, cache). loss = entropía cruzada media del lote.

    cache = (X, Z1, A1, Z2, A2, P) con P = softmax(Z3) de forma (B, 10).
    La entropía cruzada usa log-sum-exp estable.
    """
    W1, b1, W2, b2, W3, b3 = p
    Z1 = X @ W1 + b1
    A1 = xp.maximum(Z1, 0)
    Z2 = A1 @ W2 + b2
    A2 = xp.maximum(Z2, 0)
    Z3 = A2 @ W3 + b3
    m = Z3.max(axis=1, keepdims=True)
    lse = m + xp.log(xp.exp(Z3 - m).sum(axis=1, keepdims=True))      # (B,1)
    P = xp.exp(Z3 - lse)                                             # softmax
    loss = (lse[:, 0] - Z3[xp.arange(Z3.shape[0]), y]).mean()
    return loss, (X, Z1, A1, Z2, A2, P)


def predict_logits(xp, p, X):
    W1, b1, W2, b2, W3, b3 = p
    A1 = xp.maximum(X @ W1 + b1, 0)
    A2 = xp.maximum(A1 @ W2 + b2, 0)
    return A2 @ W3 + b3


# --------------------------------------------------------------------------
# 6) TODO (estudiante): backward y actualización  -- NumPy
# --------------------------------------------------------------------------
def backward_numpy(p, cache, y):
    """Regla de la cadena explícita. Devuelve [dW1, db1, dW2, db2, dW3, db3] (float32).

    Pistas: dZ3 = (P - onehot(y)) / B ; dW3 = A2.T @ dZ3 ; dA2 = dZ3 @ W3.T ;
            dZ2 = dA2 * (Z2 > 0) ; ... (no calcules dX de la primera capa).
    Prohibido: torch, tensorflow, keras, jax, autograd.
    """
    X, Z1, A1, Z2, A2, P = cache
    W1, b1, W2, b2, W3, b3 = p
    B = X.shape[0]

    # Softmax + entropía cruzada: (P - Y) / B. El /B va solo aquí; las demás capas lo heredan.
    dZ3 = P.copy()
    dZ3[np.arange(B), y] -= 1
    dZ3 /= B
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)

    # ReLU deja pasar el gradiente solo donde Z > 0.
    dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


def sgd_update_numpy(p, grads, lr):
    """Actualiza la lista p IN-PLACE: p[i] -= lr * grads[i]."""
    for w, g in zip(p, grads):
        w -= lr * g


# --------------------------------------------------------------------------
# 7) TODO (estudiante): backward y actualización  -- CuPy
# --------------------------------------------------------------------------
def backward_cupy(p, cache, y):
    """Igual que backward_numpy pero con arreglos CuPy en VRAM (sin copias a CPU)."""
    import cupy as cp                       # ya importado por el backend: solo toma el módulo en caché
    W1, b1, W2, b2, W3, b3 = p
    X, Z1, A1, Z2, A2, P = cache
    B = X.shape[0]

    # Capa de salida: softmax + entropía cruzada -> dZ3 = (P - onehot(y)) / B
    dZ3 = P.copy()                          # no tocar P del cache
    dZ3[cp.arange(B), y] -= 1               # restar el one-hot sin construir la matriz (B, 10)
    dZ3 /= B                                # el único /B: la pérdida ya viene promediada
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)

    # Capa oculta 2: ReLU
    dZ2 = (dZ3 @ W3.T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    # Capa oculta 1: ReLU (no hace falta dX)
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]   # todo float32 y en VRAM


def sgd_update_cupy(p, grads, lr):
    """Actualiza la lista p IN-PLACE en la GPU (p[i] -= lr * grads[i])."""
    for W, dW in zip(p, grads):
        W -= lr * dW                        # in-place: mismo buffer en VRAM, sin copias a CPU


# --------------------------------------------------------------------------
# 8) TODO (estudiante): backward y actualización -- PyTorch
# --------------------------------------------------------------------------
def torch_backward_and_update(loss, optimizer):
    """Calcula gradientes y actualiza con torch.optim.SGD (sin momentum)."""
    # TODO (estudiante): backward + paso del optimizador (dos líneas)
    raise NotImplementedError("Completa torch_backward_and_update")


# --------------------------------------------------------------------------
# 9) Backends con interfaz común
#    init_context -> warmup -> upload -> (set_perm + train_epoch + evaluate)* -> download_params
# --------------------------------------------------------------------------
class NumpyBackend:
    name = "numpy"

    def __init__(self, lr):
        self.lr = lr
        self.X = self.y = self.Xte = self.yte = self.p = None

    # --- ciclo de vida ---
    def init_context(self):
        pass

    def sync(self):
        pass

    def _load_params(self, params_host):
        return [a.copy() for a in params_host]

    def _step(self, Xb, yb):
        loss, cache = forward(np, self.p, Xb, yb)
        grads = backward_numpy(self.p, cache, yb)
        sgd_update_numpy(self.p, grads, self.lr)
        return loss

    def warmup(self, params_host, bs):
        """Pasos de calentamiento con datos ficticios; NO afectan al entrenamiento real."""
        rng = np.random.default_rng(0)
        self.p = self._load_params(params_host)
        Xd = rng.random((bs, ARCH[0]), dtype=np.float32)
        yd = rng.integers(0, ARCH[-1], bs)
        for _ in range(WARMUP_ITERS):
            self._step(Xd, yd)
        self.p = None

    def upload(self, Xtr, ytr, Xte, yte, params_host):
        self.X, self.y, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.p = self._load_params(params_host)
        return 0.0                                  # no hay transferencia en CPU

    def set_perm(self, perm):
        self.perm = perm
        return 0.0

    # --- entrenamiento / evaluación ---
    def train_epoch(self, bs, max_steps=None):
        n = self.X.shape[0]
        acc = np.float32(0.0)
        steps = 0
        for s in range(0, n, bs):
            idx = self.perm[s:s + bs]
            loss = self._step(self.X[idx], self.y[idx])
            acc += loss * np.float32(len(idx))      # pérdida acumulada (suma ponderada)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        return float(acc) / min(n, steps * bs)

    def evaluate(self):
        logits = predict_logits(np, self.p, self.Xte)
        return _eval_from_logits(np, logits, self.yte)

    def download_params(self):
        return [a.copy() for a in self.p], 0.0


class CupyBackend:
    name = "cupy"

    def __init__(self, lr):
        self.lr = lr
        self.p = None

    def init_context(self):
        import cupy as cp
        self.cp = cp
        cp.cuda.Device(0).use()
        cp.zeros(1).sum()                           # crea el contexto CUDA (fuera de la medición)
        cp.cuda.Stream.null.synchronize()

    def sync(self):
        self.cp.cuda.Stream.null.synchronize()

    def _load_params(self, params_host):
        return [self.cp.asarray(a) for a in params_host]

    def _step(self, Xb, yb):
        loss, cache = forward(self.cp, self.p, Xb, yb)
        grads = backward_cupy(self.p, cache, yb)
        sgd_update_cupy(self.p, grads, self.lr)
        return loss

    def warmup(self, params_host, bs):
        """Compila kernels, inicializa cuBLAS y llena el pool de memoria con las formas reales."""
        cp = self.cp
        rng = np.random.default_rng(0)
        self.p = self._load_params(params_host)
        Xd = cp.asarray(rng.random((bs, ARCH[0]), dtype=np.float32))
        yd = cp.asarray(rng.integers(0, ARCH[-1], bs))
        acc = cp.zeros((), dtype=cp.float32)
        for _ in range(WARMUP_ITERS):
            acc += self._step(Xd, yd) * np.float32(bs)
        self.sync()
        self.p = None

    def upload(self, Xtr, ytr, Xte, yte, params_host):
        """Transferencia H->D medida con sincronización explícita (pageable memory)."""
        cp = self.cp
        self.sync()
        t0 = time.perf_counter()
        self.X, self.y = cp.asarray(Xtr), cp.asarray(ytr)
        self.Xte, self.yte = cp.asarray(Xte), cp.asarray(yte)
        self.p = self._load_params(params_host)
        self.sync()
        return time.perf_counter() - t0

    def set_perm(self, perm):
        self.sync()
        t0 = time.perf_counter()
        self.perm = self.cp.asarray(perm)
        self.sync()
        return time.perf_counter() - t0

    def train_epoch(self, bs, max_steps=None):
        cp = self.cp
        n = self.X.shape[0]
        acc = cp.zeros((), dtype=cp.float32)        # pérdida acumulada EN LA GPU (sin sincronizar por iteración)
        steps = 0
        for s in range(0, n, bs):
            idx = self.perm[s:s + bs]
            loss = self._step(self.X[idx], self.y[idx])
            acc += loss * np.float32(len(idx))
            steps += 1
            if max_steps and steps >= max_steps:
                break
        return float(acc) / min(n, steps * bs)      # float() sincroniza

    def evaluate(self):
        logits = predict_logits(self.cp, self.p, self.Xte)
        return _eval_from_logits(self.cp, logits, self.yte)

    def download_params(self):
        self.sync()
        t0 = time.perf_counter()
        out = [self.cp.asnumpy(a) for a in self.p]  # D->H
        return out, time.perf_counter() - t0


class TorchBackend:
    name = "torch"

    def __init__(self, lr):
        self.lr = lr
        self.model = None

    def init_context(self):
        import torch
        self.torch = torch
        assert torch.cuda.is_available(), "PyTorch no ve ninguna GPU CUDA"
        self.dev = torch.device("cuda:0")
        torch.manual_seed(SEED)
        # float32 "de verdad": sin TF32, igual que NumPy/CuPy
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        torch.zeros(1, device=self.dev).sum()
        torch.cuda.synchronize()

    def sync(self):
        self.torch.cuda.synchronize()

    def _load_params(self, params_host):
        """Construye el modelo y copia los pesos de NumPy. Convención: nn.Linear guarda (salida, entrada)."""
        torch, nn = self.torch, self.torch.nn
        model = nn.Sequential(
            nn.Linear(ARCH[0], ARCH[1]), nn.ReLU(),
            nn.Linear(ARCH[1], ARCH[2]), nn.ReLU(),
            nn.Linear(ARCH[2], ARCH[3]),
        ).to(self.dev)
        linears = [m for m in model if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for lin, W, b in zip(linears, params_host[0::2], params_host[1::2]):
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))   # (in,out) -> (out,in)
                lin.bias.copy_(torch.from_numpy(b))
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()                                    # media del lote
        self.opt = torch.optim.SGD(model.parameters(), lr=self.lr)              # sin momentum ni decay
        return model

    def _step(self, Xb, yb):
        logits = self.model(Xb)
        loss = self.loss_fn(logits, yb)
        self.opt.zero_grad(set_to_none=True)
        torch_backward_and_update(loss, self.opt)
        return loss.detach()

    def warmup(self, params_host, bs):
        torch = self.torch
        rng = np.random.default_rng(0)
        self._load_params(params_host)
        Xd = torch.from_numpy(rng.random((bs, ARCH[0]), dtype=np.float32)).to(self.dev)
        yd = torch.from_numpy(rng.integers(0, ARCH[-1], bs)).to(self.dev)
        acc = torch.zeros((), device=self.dev)
        for _ in range(WARMUP_ITERS):
            acc += self._step(Xd, yd) * bs
        self.sync()
        self.model = None

    def upload(self, Xtr, ytr, Xte, yte, params_host):
        torch = self.torch
        self.sync()
        t0 = time.perf_counter()
        self.X = torch.from_numpy(Xtr).to(self.dev)
        self.y = torch.from_numpy(ytr).to(self.dev)
        self.Xte = torch.from_numpy(Xte).to(self.dev)
        self.yte = torch.from_numpy(yte).to(self.dev)
        self._load_params(params_host)
        self.sync()
        return time.perf_counter() - t0

    def set_perm(self, perm):
        self.sync()
        t0 = time.perf_counter()
        self.perm = self.torch.from_numpy(perm).to(self.dev)
        self.sync()
        return time.perf_counter() - t0

    def train_epoch(self, bs, max_steps=None):
        torch = self.torch
        n = self.X.shape[0]
        acc = torch.zeros((), device=self.dev)
        steps = 0
        for s in range(0, n, bs):
            idx = self.perm[s:s + bs]
            loss = self._step(self.X[idx], self.y[idx])
            acc += loss * len(idx)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        return float(acc.item()) / min(n, steps * bs)

    def evaluate(self):
        with self.torch.no_grad():
            logits = self.model(self.Xte)
            return _eval_from_logits(self.torch, logits, self.yte)

    def download_params(self):
        self.sync()
        t0 = time.perf_counter()
        lins = [m for m in self.model if isinstance(m, self.torch.nn.Linear)]
        out = []
        for lin in lins:                                                     # (out,in) -> (in,out)
            out += [lin.weight.detach().cpu().numpy().T.copy(), lin.bias.detach().cpu().numpy().copy()]
        return out, time.perf_counter() - t0


BACKENDS = {"numpy": NumpyBackend, "cupy": CupyBackend, "torch": TorchBackend}


def _eval_from_logits(lib, logits, y):
    """(pérdida media, accuracy) en test. Funciona con numpy, cupy y torch."""
    if lib is np or lib.__name__ == "cupy":
        m = logits.max(axis=1, keepdims=True)
        lse = (m + lib.log(lib.exp(logits - m).sum(axis=1, keepdims=True)))[:, 0]
        loss = (lse - logits[lib.arange(logits.shape[0]), y]).mean()
        acc = (logits.argmax(axis=1) == y).mean()
        return float(loss), float(acc)
    loss = lib.nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(dim=1) == y).float().mean()
    return float(loss.item()), float(acc.item())


# --------------------------------------------------------------------------
# 10) Información del entorno (para el reporte)
# --------------------------------------------------------------------------
def environment_info(backend):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or "desconocido",
        "numpy": np.__version__,
        "threads": thread_report(),
    }
    try:                                              # modelo de CPU legible en Linux
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu_logical_cores"] = os.cpu_count()
    if backend == "cupy":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        info.update(cupy=cp.__version__, gpu=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
                    gpu_mem_gb=round(props["totalGlobalMem"] / 2**30, 2))
    elif backend == "torch":
        import torch
        info.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                    gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    return info


# --------------------------------------------------------------------------
# 11) Experimento completo (lo que consume run_experiments.py)
# --------------------------------------------------------------------------
def run_experiment(backend, batch_size, epochs, lr=LR, synthetic=False, train_limit=None,
                   test_limit=None, eval_every=1):
    """Devuelve dict con transfer_s, compute_s, total_s, test_acc, loss_por_época y detalles.

    Definiciones (úsalas igual en el reporte):
      transfer_s = H->D (datos, pesos y permutaciones) + D->H (pesos finales)
      compute_s  = suma del tiempo de entrenamiento por época (sin evaluación, con sync)
      total_s    = transfer_s + compute_s
      eval_s     = tiempo de evaluación en test (se reporta aparte; NO entra en total_s)
    El calentamiento y la creación del contexto CUDA quedan fuera de toda medición.
    """
    Xtr, ytr, Xte, yte = load_mnist(synthetic, train_limit, test_limit)
    params_host = he_init()
    B = BACKENDS[backend](lr)
    B.init_context()
    B.warmup(params_host, batch_size)                       # fuera de la medición
    h2d_s = B.upload(Xtr, ytr, Xte, yte, params_host)

    n = Xtr.shape[0]
    epoch_times, train_loss, test_loss, test_acc, cum_compute = [], [], [], [], []
    eval_s, compute_s = 0.0, 0.0
    for ep in range(epochs):
        h2d_s += B.set_perm(epoch_permutation(ep, n))
        B.sync()
        t0 = time.perf_counter()
        tl = B.train_epoch(batch_size)
        B.sync()
        dt = time.perf_counter() - t0
        compute_s += dt
        epoch_times.append(dt)
        cum_compute.append(compute_s)
        train_loss.append(tl)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            te0 = time.perf_counter()
            l, a = B.evaluate()
            B.sync()
            eval_s += time.perf_counter() - te0
            test_loss.append(l)
            test_acc.append(a)
        else:
            test_loss.append(None)
            test_acc.append(None)
    _, d2h_s = B.download_params()
    transfer_s = h2d_s + d2h_s
    return {
        "backend": backend,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "n_train": int(n),
        "iters_per_epoch": int(-(-n // batch_size)),
        "synthetic": bool(synthetic),
        "transfer_s": transfer_s,
        "h2d_s": h2d_s,
        "d2h_s": d2h_s,
        "compute_s": compute_s,
        "total_s": transfer_s + compute_s,
        "eval_s": eval_s,
        "test_acc": test_acc[-1],
        "loss_por_época": train_loss,
        "test_loss_por_época": test_loss,
        "test_acc_por_época": test_acc,
        "epoch_times_s": epoch_times,
        "cum_compute_s": cum_compute,
        "env": environment_info(backend),
    }


# --------------------------------------------------------------------------
# 12) Verificación de gradientes por diferencias finitas (NumPy, red pequeña, float64)
# --------------------------------------------------------------------------
def gradient_check(eps=1e-5, tol=1e-6, arch=(20, 8, 6, 4), batch=7, seed=0):
    """Compara backward_numpy contra diferencias finitas centrales. Devuelve True si pasa."""
    rng = np.random.default_rng(seed)
    p = he_init(arch, seed=seed, dtype=np.float64)
    p = [a + (0.1 * rng.standard_normal(a.shape) if a.ndim == 1 else 0.0) for a in p]   # b != 0 para probar db
    X = rng.standard_normal((batch, arch[0]))
    y = rng.integers(0, arch[-1], batch)
    _, cache = forward(np, p, X, y)
    grads = backward_numpy(p, cache, y)
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    ok = True
    print(f"{'param':<5} {'err. relativo máx.':>20}   estado")
    for k, (a, g) in enumerate(zip(p, grads)):
        num = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps
            lp, _ = forward(np, p, X, y)
            a[i] = old - eps
            lm, _ = forward(np, p, X, y)
            a[i] = old
            num[i] = (lp - lm) / (2 * eps)
        err = np.max(np.abs(g - num) / np.maximum(np.abs(g) + np.abs(num), 1e-8))
        good = err < tol
        ok &= bool(good)
        print(f"{names[k]:<5} {err:>20.3e}   {'OK' if good else 'FALLA'}")
    print("gradient_check:", "APROBADO" if ok else "REPROBADO", f"(tol={tol:g}, eps={eps:g}, float64)")
    return ok


# --------------------------------------------------------------------------
# 13) Consistencia entre backends: mismos pesos tras unos pasos
# --------------------------------------------------------------------------
def consistency_check(backends=("numpy", "cupy", "torch"), bs=128, steps=3, synthetic=False, tol=1e-4):
    """Entrena `steps` iteraciones con los mismos lotes y compara pesos contra NumPy.

    Detecta errores típicos: convención (in,out) vs (out,in) en PyTorch, signo del gradiente,
    olvidar dividir entre B, etc.
    """
    Xtr, ytr, Xte, yte = load_mnist(synthetic, train_limit=2048, test_limit=512)
    params_host = he_init()
    results = {}
    for name in backends:
        try:
            B = BACKENDS[name](LR)
            B.init_context()
            B.upload(Xtr, ytr, Xte, yte, params_host)
            B.set_perm(epoch_permutation(0, Xtr.shape[0]))
            B.train_epoch(bs, max_steps=steps)
            B.sync()
            results[name], _ = B.download_params()
        except NotImplementedError as e:
            print(f"[{name}] pendiente: {e}")
        except Exception as e:                       # p. ej. no hay GPU/CuPy/PyTorch
            print(f"[{name}] no disponible: {type(e).__name__}: {e}")
    if "numpy" not in results:
        print("Sin referencia NumPy no hay comparación.")
        return False
    ok = True
    for name, params in results.items():
        if name == "numpy":
            continue
        diff = max(float(np.max(np.abs(a - b))) for a, b in zip(results["numpy"], params))
        good = diff < tol
        ok &= good
        print(f"max|W_numpy - W_{name}| tras {steps} pasos = {diff:.3e}  ->  {'OK' if good else 'REVISAR'} (tol={tol:g})")
    return ok


# --------------------------------------------------------------------------
# 14) CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS), help="numpy | cupy | torch")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true", help="datos sintéticos (solo prueba de flujo)")
    ap.add_argument("--train-limit", type=int, default=None, help="usar solo los primeros N de train (prueba rápida)")
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="guardar el resultado en JSON")
    ap.add_argument("--gradcheck", action="store_true", help="verificación de gradientes (NumPy)")
    ap.add_argument("--consistency", action="store_true", help="comparar pesos de los backends tras 3 pasos")
    args = ap.parse_args()

    if args.gradcheck:
        sys.exit(0 if gradient_check() else 1)
    if args.consistency:
        sys.exit(0 if consistency_check(synthetic=args.synthetic) else 1)
    if not args.backend:
        ap.error("indica --backend (o usa --gradcheck / --consistency)")

    res = run_experiment(args.backend, args.batch_size, args.epochs, args.lr, args.synthetic,
                         args.train_limit, args.test_limit, args.eval_every)
    summary = {k: res[k] for k in ("backend", "batch_size", "epochs", "transfer_s", "compute_s",
                                    "total_s", "eval_s", "test_acc", "loss_por_época")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
