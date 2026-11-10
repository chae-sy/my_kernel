
import torch
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# 1. Triton kernel
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# 2. Python wrapper
def add(x: torch.Tensor, y: torch.Tensor):
    assert x.device == DEVICE and y.device == DEVICE
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()

    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=1024,
    )

    return output


# 3. Numerical equivalence test
def test_correctness():
    print("=== Numerical Equivalence Test ===")

    torch.manual_seed(0)
    size = 98432

    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    output_torch = x + y
    output_triton = add(x, y)

    max_diff = torch.max(
        torch.abs(output_torch - output_triton)
    ).item()

    print(f"Size: {size}")
    print(f"Maximum difference: {max_diff:.10e}")

    torch.testing.assert_close(
        output_triton,
        output_torch,
        rtol=1e-5,
        atol=1e-6,
    )

    print("Numerical equivalence: PASSED\n")


# 4. Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y,
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gbps = lambda ms: (
        3 * x.numel() * x.element_size() * 1e-9
        / (ms * 1e-3)
    )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# 5. Main
if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    test_correctness()

    print("=== Performance Benchmark ===")
    # vector_add.py가 위치한 디렉터리 기준
    save_path = Path(__file__).resolve().parent / "benchmark_results"
    save_path.mkdir(parents=True, exist_ok=True)
    benchmark.run(
        print_data=True,
        show_plots=True,
        save_path=str(save_path),
    )
