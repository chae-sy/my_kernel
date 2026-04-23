import torch
import time
import runner

torch.manual_seed(0)

M = N = K = 4096
WARMUP = 100
REPS = 100
GFLOPs = 2 * M * N * K / 1e9

A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C0 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
C1 = torch.zeros_like(C0)
C2 = torch.zeros_like(C0)
C3 = torch.zeros_like(C0)
C4 = torch.zeros_like(C0)

def run_benchmark(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times)

def compare(C_ref, C_test):
    diff = (C_ref - C_test).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    mism = (diff.abs() > 5e-2).sum().item()
    return max_abs, mean_abs, mism

print(f"\n=== Benchmark: M={M}, N={N}, K={K} ===")
print(f"Warm-up {WARMUP} iters, Measure {REPS} iters\n")

# torch.matmul baseline
t_torch = run_benchmark(lambda: torch.matmul(A, B.T))
C_ref = torch.matmul(A, B.T)
gflops_torch = GFLOPs / (t_torch / 1e3)

# === kernels ===
# runner.matmul_0_1(A, B, C0)  # optional baseline

t_10 = run_benchmark(lambda: runner.mma_matmul_1_0(A, B, C1))
t_11 = run_benchmark(lambda: runner.mma_matmul_1_1(A, B, C2))
t_20 = run_benchmark(lambda: runner.mma_matmul_2_0(A, B, C3))
t_21 = run_benchmark(lambda: runner.mma_matmul_2_1(A, B, C4))

# === accuracy ===
max10, mean10, mism10 = compare(C_ref, C1)
max11, mean11, mism11 = compare(C_ref, C2)
max20, mean20, mism20 = compare(C_ref, C3)
max21, mean21, mism21 = compare(C_ref, C4)

# === performance ===
gflops_10 = GFLOPs / (t_10 / 1e3)
gflops_11 = GFLOPs / (t_11 / 1e3)
gflops_20 = GFLOPs / (t_20 / 1e3)
gflops_21 = GFLOPs / (t_21 / 1e3)

r10 = 100 * gflops_10 / gflops_torch
r11 = 100 * gflops_11 / gflops_torch
r20 = 100 * gflops_20 / gflops_torch
r21 = 100 * gflops_21 / gflops_torch

# === print results ===
print(f"=== Accuracy ===")
print(f"[mma_matmul_1_0]  max {max10:.6f}, mean {mean10:.6f}, mism {mism10}")
print(f"[mma_matmul_1_1]  max {max11:.6f}, mean {mean11:.6f}, mism {mism11}")
print(f"[mma_matmul_2_0]  max {max20:.6f}, mean {mean20:.6f}, mism {mism20}")
print(f"[mma_matmul_2_1]  max {max21:.6f}, mean {mean21:.6f}, mism {mism21}")

print(f"\n=== Performance (avg of {REPS} runs after {WARMUP} warmup) ===")
print(f"{'Kernel':<15} {'Time (ms)':>12} {'GFLOP/s':>12} {'Rel. Perf':>12}")
print(f"{'-'*55}")
print(f"{'torch.matmul':<15} {t_torch:>12.4f} {gflops_torch:>12.2f} {'100.00%':>12}")
print(f"{'mma_matmul_1_0':<15} {t_10:>12.4f} {gflops_10:>12.2f} {r10:>11.2f}%")
print(f"{'mma_matmul_1_1':<15} {t_11:>12.4f} {gflops_11:>12.2f} {r11:>11.2f}%")
print(f"{'mma_matmul_2_0':<15} {t_20:>12.4f} {gflops_20:>12.2f} {r20:>11.2f}%")
print(f"{'mma_matmul_2_1':<15} {t_21:>12.4f} {gflops_21:>12.2f} {r21:>11.2f}%")

print("\nDone ✅")
