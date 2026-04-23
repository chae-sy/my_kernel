from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='runner',  # ⚠️ 반드시 PYBIND11_MODULE 이름과 같아야 함
    ext_modules=[
        CUDAExtension(
            name='runner',
            sources=['runner.cu', 'kernel.cu'],  # entry + kernel
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__'
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
