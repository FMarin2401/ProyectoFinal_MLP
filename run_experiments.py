#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments.py  --  Barrido de backends x batch sizes para el Proyecto Final (MLP en MNIST)

Cada corrida se lanza en un SUBPROCESO LIMPIO (hilos de CPU = 1, sin estado compartido) con
proyecto_mlp_mnist.py y deja un JSON en <out-dir>/raw/. Después se generan CSV y gráficas.

Protocolos (ver enunciado, sección 4):
  T  (tiempos)      : backends x batch {32,128,512,2048}, pocas épocas (por defecto 3),
                      3 repeticiones en GPU y 1 en NumPy (configurable).
  C  (convergencia) : batch 128 en los tres backends, 20 épocas, 1 repetición, evaluación en test
                      en cada época (fuera del cronómetro). Alcance completo: --conv-batches 32 128 512 2048
                      (los batches distintos de --conv-numpy-batch se corren solo en GPU).

Uso:
  python run_experiments.py --mode quick                 # prueba de flujo (minutos)
  python run_experiments.py --mode quick --synthetic     # idem, sin internet y sin MNIST
  python run_experiments.py --mode full                  # barrido completo
  python run_experiments.py --mode full --resume         # retoma sin repetir corridas ya hechas
  python run_experiments.py --analyze-only --peak-gflops 8100   # solo CSV y gráficas

Salidas (<out-dir>): raw/*.json, logs/*.log, timing.csv, summary.csv, convergence.csv,
time_to_acc.csv, env.json, hardware_peak.json, fig_*.png, failures.log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAYERS = (784, 512, 256, 10)
GPU_BACKENDS = ("cupy", "torch")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# --------------------------------------------------------------------------
# FLOPs (convención: solo GEMM, 2 FLOP por multiplicación-acumulación)
# Debes DERIVARLO a mano en el reporte y compararlo con esta función.
# --------------------------------------------------------------------------
def flops_per_sample(layers=LAYERS):
    pairs = list(zip(layers[:-1], layers[1:]))
    fwd = sum(2 * i * o for i, o in pairs)                  # Z = A @ W
    bwd_dw = sum(2 * i * o for i, o in pairs)               # dW = A.T @ dZ
    bwd_dx = sum(2 * i * o for i, o in pairs[1:])           # dA = dZ @ W.T (la capa 1 no lo necesita)
    return fwd + bwd_dw + bwd_dx


def flops_per_iteration(bs, layers=LAYERS):
    return bs * flops_per_sample(layers)


# --------------------------------------------------------------------------
# Lanzamiento de corridas
# --------------------------------------------------------------------------
def clean_env():
    env = os.environ.copy()
    for v in THREAD_VARS:
        env[v] = "1"
    env["CUPY_TF32"] = "0"
    env["PYTHONHASHSEED"] = "0"
    return env


def run_one(script, backend, bs, epochs, proto, rep, out_dir, eval_every, args):
    raw = out_dir / "raw"
    logs = out_dir / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tag = f"{proto}_{backend}_bs{bs}_r{rep}"
    out_json, log = raw / f"{tag}.json", logs / f"{tag}.log"
    if args.resume and out_json.exists():
        print(f"  [salta] {tag} (ya existe)")
        return True
    cmd = [sys.executable, str(script), "--backend", backend, "--batch-size", str(bs),
           "--epochs", str(epochs), "--eval-every", str(eval_every), "--out", str(out_json)]
    if args.synthetic:
        cmd.append("--synthetic")
    if args.train_limit:
        cmd += ["--train-limit", str(args.train_limit)]
    if args.test_limit:
        cmd += ["--test-limit", str(args.test_limit)]
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=clean_env(), cwd=str(HERE))
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not out_json.exists():
        tail = "".join(open(log, encoding="utf-8", errors="replace").readlines()[-6:])
        with open(out_dir / "failures.log", "a", encoding="utf-8") as ff:
            ff.write(f"=== {tag} (código {proc.returncode}) ===\n{tail}\n")
        print(f"  [FALLÓ] {tag}: {tail.strip().splitlines()[-1] if tail.strip() else 'sin salida'}")
        return False
    res = json.load(open(out_json, encoding="utf-8"))
    print(f"  [ok] {tag}: compute={res['compute_s']:.2f}s transfer={res['transfer_s']:.2f}s "
          f"acc={res['test_acc']:.4f} (pared {dt:.0f}s)")
    return True


def run_gemm_benchmark(out_dir, batch_sizes):
    """Mide el GFLOP/s alcanzable de un SGEMM cuBLAS (referencia de 'eficiencia'). Subproceso aparte."""
    out = out_dir / "hardware_peak.json"
    code = r"""
import json, sys, time
bss = json.loads(sys.argv[1]); res = {}
try:
    import cupy as cp
    def mk(m, k, n):
        return cp.random.rand(m, k, dtype=cp.float32), cp.random.rand(k, n, dtype=cp.float32)
    def sync(): cp.cuda.Stream.null.synchronize()
    lib = "cupy"; mm = lambda a, b: a @ b
except Exception:
    import torch
    def mk(m, k, n):
        return torch.rand(m, k, device="cuda"), torch.rand(k, n, device="cuda")
    def sync(): torch.cuda.synchronize()
    lib = "torch"; mm = lambda a, b: a @ b
    torch.backends.cuda.matmul.allow_tf32 = False
def bench(m, k, n, reps):
    a, b = mk(m, k, n)
    for _ in range(5): mm(a, b)
    sync(); t0 = time.perf_counter()
    for _ in range(reps): mm(a, b)
    sync(); dt = (time.perf_counter() - t0) / reps
    return 2.0 * m * k * n / dt / 1e9
res["library"] = lib
res["gemm_large_gflops"] = bench(4096, 4096, 4096, 20)
res["gemm_shape_gflops"] = {str(b): bench(b, 784, 512, 200) for b in bss}
json.dump(res, open(sys.argv[2], "w"), indent=2)
"""
    proc = subprocess.run([sys.executable, "-c", code, json.dumps(batch_sizes), str(out)],
                          capture_output=True, text=True, env=clean_env())
    if proc.returncode != 0:
        print("  [GEMM] no disponible:", (proc.stderr.strip().splitlines() or ["?"])[-1])
        return None
    d = json.load(open(out))
    print(f"  [GEMM] SGEMM 4096^3 ({d['library']}): {d['gemm_large_gflops']:.0f} GFLOP/s")
    return d


# --------------------------------------------------------------------------
# Análisis: CSV + gráficas
# --------------------------------------------------------------------------
def load_raw(out_dir, proto):
    rows = []
    for f in sorted((out_dir / "raw").glob(f"{proto}_*.json")):
        r = json.load(open(f, encoding="utf-8"))
        r["file"] = f.name
        r["rep"] = int(f.stem.rsplit("_r", 1)[1])
        rows.append(r)
    return rows


def analyze(out_dir, peak_gflops=None, target_acc=0.95):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    gemm = json.load(open(out_dir / "hardware_peak.json")) if (out_dir / "hardware_peak.json").exists() else {}
    gemm_large = gemm.get("gemm_large_gflops")

    # ---------- Protocolo T ----------
    T = load_raw(out_dir, "T")
    if T:
        df = pd.DataFrame([{
            "backend": r["backend"], "batch_size": r["batch_size"], "rep": r["rep"], "epochs": r["epochs"],
            "n_train": r["n_train"], "iters_per_epoch": r["iters_per_epoch"],
            "h2d_s": r["h2d_s"], "d2h_s": r["d2h_s"], "transfer_s": r["transfer_s"],
            "compute_s": r["compute_s"], "total_s": r["total_s"], "eval_s": r["eval_s"],
            "test_acc": r["test_acc"], "final_train_loss": r["loss_por_época"][-1],
        } for r in T])
        df["flops_total"] = df.apply(lambda x: x.epochs * x.n_train * flops_per_sample(), axis=1)
        df["gflops_compute"] = df.flops_total / df.compute_s / 1e9
        df["gflops_total"] = df.flops_total / df.total_s / 1e9
        df.sort_values(["backend", "batch_size", "rep"]).to_csv(out_dir / "timing.csv", index=False)

        g = df.groupby(["backend", "batch_size"])
        summ = g.agg(reps=("rep", "count"),
                     transfer_mean=("transfer_s", "mean"), transfer_std=("transfer_s", "std"),
                     compute_mean=("compute_s", "mean"), compute_std=("compute_s", "std"),
                     total_mean=("total_s", "mean"), total_std=("total_s", "std"),
                     h2d_mean=("h2d_s", "mean"), d2h_mean=("d2h_s", "mean"),
                     test_acc=("test_acc", "mean"), gflops_compute=("gflops_compute", "mean"),
                     gflops_total=("gflops_total", "mean")).reset_index()
        base = summ[summ.backend == "numpy"].set_index("batch_size")
        summ["speedup_compute"] = summ.apply(
            lambda x: base.compute_mean.get(x.batch_size, np.nan) / x.compute_mean if x.backend != "numpy" else np.nan, axis=1)
        summ["speedup_total"] = summ.apply(
            lambda x: base.total_mean.get(x.batch_size, np.nan) / x.total_mean if x.backend != "numpy" else np.nan, axis=1)
        if gemm_large:
            summ["eff_vs_gemm_pct"] = 100.0 * summ.gflops_compute / gemm_large
        if peak_gflops:
            summ["eff_vs_peak_pct"] = 100.0 * summ.gflops_compute / peak_gflops
        summ.to_csv(out_dir / "summary.csv", index=False)
        print("\n== Resumen (summary.csv) ==")
        print(summ.round(3).to_string(index=False))

        # entorno
        env = {}
        for r in T:
            env.setdefault(r["backend"], r["env"])
        json.dump(env, open(out_dir / "env.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        bss = sorted(summ.batch_size.unique())
        colors = {"numpy": "tab:gray", "cupy": "tab:blue", "torch": "tab:red"}

        # Fig 1: desglose transferencia vs cómputo + total (log)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        w, x = 0.38, np.arange(len(bss))
        for k, b in enumerate(GPU_BACKENDS):
            s = summ[summ.backend == b].set_index("batch_size").reindex(bss)
            if s.compute_mean.isna().all():
                continue
            xs = x + (k - 0.5) * w
            ax[0].bar(xs, s.compute_mean, w, label=f"{b}: cómputo", color=colors[b])
            ax[0].bar(xs, s.transfer_mean, w, bottom=s.compute_mean, label=f"{b}: transferencia H↔D",
                      color=colors[b], alpha=0.45, hatch="//")
        ax[0].set_xticks(x, bss)
        ax[0].set_xlabel("batch size")
        ax[0].set_ylabel("segundos (medias)")
        ax[0].set_title("GPU: cómputo vs transferencia")
        if ax[0].get_legend_handles_labels()[0]:
            ax[0].legend(fontsize=8)
        for b in ("numpy",) + GPU_BACKENDS:
            s = summ[summ.backend == b].sort_values("batch_size")
            if len(s):
                ax[1].plot(s.batch_size, s.total_mean, "o-", color=colors[b], label=b)
        ax[1].set_xscale("log", base=2)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("batch size")
        ax[1].set_ylabel("tiempo total (s, log)")
        ax[1].set_title("Tiempo total por backend")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_desglose.png", dpi=150)
        plt.close(fig)

        # Fig 2: speedup vs batch
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for b in GPU_BACKENDS:
            s = summ[summ.backend == b].sort_values("batch_size")
            if len(s) and s.speedup_compute.notna().any():
                ax.plot(s.batch_size, s.speedup_compute, "o-", color=colors[b], label=f"{b} (cómputo)")
                ax.plot(s.batch_size, s.speedup_total, "s--", color=colors[b], label=f"{b} (total con transferencia)")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("S = T_numpy / T_gpu")
        ax.set_title("Speedup frente a NumPy (1 hilo)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_speedup.png", dpi=150)
        plt.close(fig)

        # Fig 3: GFLOP/s
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for b in ("numpy",) + GPU_BACKENDS:
            s = summ[summ.backend == b].sort_values("batch_size")
            if len(s):
                ax.plot(s.batch_size, s.gflops_compute, "o-", color=colors[b], label=b)
        if gemm_large:
            ax.axhline(gemm_large, color="k", ls=":", label=f"SGEMM 4096³ medido ({gemm_large:.0f})")
        if peak_gflops:
            ax.axhline(peak_gflops, color="k", ls="--", label=f"pico teórico ({peak_gflops:.0f})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("GFLOP/s (solo GEMM, cómputo puro)")
        ax.set_title("Rendimiento alcanzado")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_gflops.png", dpi=150)
        plt.close(fig)
    else:
        print("No hay corridas del protocolo T en", out_dir / "raw")

    # ---------- Protocolo C ----------
    C = load_raw(out_dir, "C")
    if C:
        rows, tta = [], []
        for r in C:
            for e in range(r["epochs"]):
                rows.append({"backend": r["backend"], "batch_size": r["batch_size"], "rep": r["rep"], "epoch": e + 1,
                             "train_loss": r["loss_por_época"][e], "test_loss": r["test_loss_por_época"][e],
                             "test_acc": r["test_acc_por_época"][e], "cum_compute_s": r["cum_compute_s"][e]})
            hit = next((e for e, a in enumerate(r["test_acc_por_época"]) if a is not None and a >= target_acc), None)
            tta.append({"backend": r["backend"], "batch_size": r["batch_size"], "rep": r["rep"], "target_acc": target_acc,
                        "epoch_reached": None if hit is None else hit + 1,
                        "compute_s_to_target": None if hit is None else r["cum_compute_s"][hit],
                        "final_test_acc": r["test_acc"]})
        dc = pd.DataFrame(rows)
        dc.to_csv(out_dir / "convergence.csv", index=False)
        pd.DataFrame(tta).to_csv(out_dir / "time_to_acc.csv", index=False)
        print(f"\n== Tiempo hasta {target_acc:.0%} de accuracy en test (time_to_acc.csv) ==")
        print(pd.DataFrame(tta).to_string(index=False))

        colors = {"numpy": "tab:gray", "cupy": "tab:blue", "torch": "tab:red"}
        ref_bs = 128 if (dc.batch_size == 128).any() else int(dc.batch_size.iloc[0])
        d = dc[dc.batch_size == ref_bs]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for b, s in d.groupby("backend"):
            ax[0].plot(s.epoch, s.train_loss, "o-", ms=3, color=colors.get(b), label=b)
            ax[1].plot(s.epoch, s.test_acc, "o-", ms=3, color=colors.get(b), label=b)
            ax[2].plot(s.cum_compute_s, s.test_acc, "o-", ms=3, color=colors.get(b), label=b)
        ax[0].set(xlabel="época", ylabel="pérdida de entrenamiento", title=f"Pérdida (batch {ref_bs})")
        ax[1].set(xlabel="época", ylabel="accuracy en test", title=f"Accuracy (batch {ref_bs})")
        ax[2].set(xlabel="tiempo de cómputo acumulado (s, log)", ylabel="accuracy en test", xscale="log",
                  title="Accuracy vs tiempo")
        for a in ax:
            a.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_convergencia.png", dpi=150)
        plt.close(fig)

        if dc.batch_size.nunique() > 1:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            styles = {"numpy": ":", "cupy": "-", "torch": "--"}
            for (b, bs), s in dc.groupby(["backend", "batch_size"]):
                ax[0].plot(s.epoch, s.test_acc, styles.get(b, "-"), label=f"{b} bs={bs}")
                ax[1].plot(s.cum_compute_s, s.test_acc, styles.get(b, "-"), label=f"{b} bs={bs}")
            ax[0].set(xlabel="época", ylabel="accuracy en test", title="Accuracy vs época según batch")
            ax[1].set(xlabel="tiempo de cómputo acumulado (s, log)", xscale="log", title="Accuracy vs tiempo según batch")
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_convergencia_batches.png", dpi=150)
            plt.close(fig)
    else:
        print("No hay corridas del protocolo C en", out_dir / "raw")
    print(f"\nListo. Archivos en: {out_dir}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["quick", "full"], default="full")
    ap.add_argument("--backends", nargs="+", default=["numpy", "cupy", "torch"], choices=["numpy", "cupy", "torch"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--epochs-timing", type=int, default=None, help="épocas del protocolo T (full: 3)")
    ap.add_argument("--epochs-conv", type=int, default=None, help="épocas del protocolo C (full: 20)")
    ap.add_argument("--reps-gpu", type=int, default=None, help="repeticiones GPU (full: 3)")
    ap.add_argument("--reps-numpy", type=int, default=None, help="repeticiones NumPy (full: 1; ideal 2-3 si el tiempo alcanza)")
    ap.add_argument("--conv-batches", nargs="+", type=int, default=None, help="batches del protocolo C (full: 128)")
    ap.add_argument("--conv-numpy-batch", type=int, default=128, help="único batch de convergencia que se corre en NumPy")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-convergence", action="store_true")
    ap.add_argument("--skip-gemm", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--script", default=str(HERE / "proyecto_mlp_mnist.py"))
    ap.add_argument("--synthetic", action="store_true", help="datos sintéticos (solo prueba de flujo)")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="no repetir corridas con JSON existente")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--peak-gflops", type=float, default=None, help="pico FP32 teórico de tu GPU (hoja de datos)")
    ap.add_argument("--target-acc", type=float, default=0.95, help="umbral de accuracy para tiempo-hasta-X%%")
    args = ap.parse_args()

    quick = args.mode == "quick"
    if quick:                                    # valores por defecto de la prueba de flujo
        args.batch_sizes = args.batch_sizes or [128, 2048]
        args.epochs_timing = args.epochs_timing or 1
        args.epochs_conv = args.epochs_conv or 2
        args.reps_gpu = args.reps_gpu or 1
        args.reps_numpy = args.reps_numpy or 1
        args.conv_batches = args.conv_batches or [128]
        args.train_limit = args.train_limit or 6000
        args.test_limit = args.test_limit or 2000
        args.target_acc = min(args.target_acc, 0.5)
    else:
        args.batch_sizes = args.batch_sizes or [32, 128, 512, 2048]
        args.epochs_timing = args.epochs_timing or 3
        args.epochs_conv = args.epochs_conv or 20
        args.reps_gpu = args.reps_gpu or 3
        args.reps_numpy = args.reps_numpy or 1
        args.conv_batches = args.conv_batches or [128]
    out_dir = Path(args.out_dir or ("results_quick" if quick else "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        dead = set()
        if not args.skip_gemm and any(b in GPU_BACKENDS for b in args.backends):
            print("== Referencia GEMM (cuBLAS) ==")
            run_gemm_benchmark(out_dir, args.batch_sizes)

        if not args.skip_timing:
            print(f"\n== Protocolo T: {args.epochs_timing} época(s), batches {args.batch_sizes} ==")
            for rep in range(1, max(args.reps_gpu, args.reps_numpy) + 1):      # repeticiones en el lazo externo
                for bs in args.batch_sizes:
                    for b in args.backends:
                        reps = args.reps_gpu if b in GPU_BACKENDS else args.reps_numpy
                        if rep > reps or b in dead:
                            continue
                        ok = run_one(args.script, b, bs, args.epochs_timing, "T", rep, out_dir,
                                     args.epochs_timing, args)
                        if not ok:
                            dead.add(b)
                            print(f"  >> se omite '{b}' en el resto del barrido (revisa failures.log)")

        if not args.skip_convergence:
            print(f"\n== Protocolo C: {args.epochs_conv} épocas, batches {args.conv_batches} ==")
            for bs in args.conv_batches:
                for b in args.backends:
                    if b in dead or (b == "numpy" and bs != args.conv_numpy_batch):
                        continue
                    ok = run_one(args.script, b, bs, args.epochs_conv, "C", 1, out_dir, 1, args)
                    if not ok:
                        dead.add(b)
        if dead:
            print(f"\nBackends con fallos: {sorted(dead)}. Detalle en {out_dir / 'failures.log'}")

    analyze(out_dir, args.peak_gflops, args.target_acc)


if __name__ == "__main__":
    main()
