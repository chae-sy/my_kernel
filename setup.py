from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='runner',
    ext_modules=[
        CUDAExtension(
            name='runner',
            sources=['runner.cu', 'kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-lcublas'
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
