import torch
import time
import runner

torch.manual_seed(0)

# ============================================================
# 1. 설정
# ============================================================
M = N = K = 256
WARMUP = 100
REPS = 100

A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')
C0 = torch.zeros(M, N, dtype=torch.float16, device='cuda')
C1 = torch.zeros_like(C0)
C2 = torch.zeros_like(C0)

# ============================================================
# 2. 헬퍼 함수
# ============================================================
def run_benchmark(fn, label):
    # warm-up
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # measure
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    return avg_ms

def compare(C_ref, C_test):
    diff = (C_ref - C_test).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    mism = (diff.abs() > 5e-2).sum().item()
    return max_abs, mean_abs, mism

GFLOPs = 2 * M * N * K / 1e9  # for 1 iteration

# ============================================================
# 3. 워밍업 후 성능 측정
# ============================================================
print(f"\n=== Benchmark: M={M}, N={N}, K={K} ===")
print(f"Warm-up {WARMUP} iters, Measure {REPS} iters\n")

# --- torch.matmul ---
t_torch = run_benchmark(lambda: torch.matmul(A, B), "torch.matmul")
C_ref = torch.matmul(A, B)

# --- kernel 1.0 ---
t_10 = run_benchmark(lambda: runner.mma_matmul_1_0(A, B, C1), "mma_matmul_1_0")

# --- kernel 1.1 ---
t_11 = run_benchmark(lambda: runner.mma_matmul_1_1(A, B, C2), "mma_matmul_1_1")

# ============================================================
# 4. 정확도 및 성능 계산
# ============================================================
max0, mean0, mism0 = compare(C_ref, C1)
max1, mean1, mism1 = compare(C_ref, C2)

gflops_torch = GFLOPs / (t_torch / 1e3)
gflops_10 = GFLOPs / (t_10 / 1e3)
gflops_11 = GFLOPs / (t_11 / 1e3)

r10 = 100 * gflops_10 / gflops_torch
r11 = 100 * gflops_11 / gflops_torch

# ============================================================
# 5. 결과 출력
# ============================================================
print("=== Accuracy ===")
print(f"[mma_matmul_1_0] max abs diff {max0:.6f}, mean abs diff {mean0:.6f}, mismatches {mism0}")
print(f"[mma_matmul_1_1] max abs diff {max1:.6f}, mean abs diff {mean1:.6f}, mismatches {mism1}")

print(f"\n=== Performance (avg of {REPS} runs after {WARMUP} warmup) ===")
print(f"{'Kernel':<15} {'Time (ms)':>12} {'GFLOP/s':>12} {'Rel. Perf':>12}")
print(f"{'-'*55}")
print(f"{'torch.matmul':<15} {t_torch:>12.4f} {gflops_torch:>12.2f} {'100.00%':>12}")
print(f"{'mma_matmul_1_0':<15} {t_10:>12.4f} {gflops_10:>12.2f} {r10:>11.2f}%")
print(f"{'mma_matmul_1_1':<15} {t_11:>12.4f} {gflops_11:>12.2f} {r11:>11.2f}%")

print("\nDone ✅")
