import torch
import time
import runner
print(dir(runner))

torch.manual_seed(0)

M = N = K = 4096
WARMUP = 100
REPS = 100

A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C0 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
C1 = torch.zeros_like(C0)
C2 = torch.zeros_like(C0)
C3 = torch.zeros_like(C0)

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

GFLOPs = 2 * M * N * K / 1e9

print(f"\n=== Benchmark: M={M}, N={N}, K={K} ===")
print(f"Warm-up {WARMUP} iters, Measure {REPS} iters\n")

# torch.matmul
t_torch = run_benchmark(lambda: torch.matmul(A, B.T))
C_ref = torch.matmul(A, B.T)

# matmul_0_1
#t_01 = run_benchmark(lambda: runner.matmul_0_1(A, B, C0))

# mma kernels
t_10 = run_benchmark(lambda: runner.mma_matmul_1_0(A, B, C1))
t_11 = run_benchmark(lambda: runner.mma_matmul_1_1(A, B, C2))

max01, mean01, mism01 = compare(C_ref, C0)
max10, mean10, mism10 = compare(C_ref, C1)
max11, mean11, mism11 = compare(C_ref, C2)

gflops_torch = GFLOPs / (t_torch / 1e3)
#gflops_01 = GFLOPs / (t_01 / 1e3)
gflops_10 = GFLOPs / (t_10 / 1e3)
gflops_11 = GFLOPs / (t_11 / 1e3)

#r01 = 100 * gflops_01 / gflops_torch
r10 = 100 * gflops_10 / gflops_torch
r11 = 100 * gflops_11 / gflops_torch

print(f"=== Accuracy ===")
print(f"[matmul_0_1]  max {max01:.6f}, mean {mean01:.6f}, mism {mism01}")
print(f"[mma_matmul_1_0]  max {max10:.6f}, mean {mean10:.6f}, mism {mism10}")
print(f"[mma_matmul_1_1]  max {max11:.6f}, mean {mean11:.6f}, mism {mism11}")

print(f"\n=== Performance (avg of {REPS} runs after {WARMUP} warmup) ===")
print(f"{'Kernel':<15} {'Time (ms)':>12} {'GFLOP/s':>12} {'Rel. Perf':>12}")
print(f"{'-'*55}")
print(f"{'torch.matmul':<15} {t_torch:>12.4f} {gflops_torch:>12.2f} {'100.00%':>12}")
#print(f"{'matmul_0_1':<15} {t_01:>12.4f} {gflops_01:>12.2f} {r01:>11.2f}%")
print(f"{'mma_matmul_1_0':<15} {t_10:>12.4f} {gflops_10:>12.2f} {r10:>11.2f}%")
print(f"{'mma_matmul_1_1':<15} {t_11:>12.4f} {gflops_11:>12.2f} {r11:>11.2f}%")

print("\nDone ✅")
