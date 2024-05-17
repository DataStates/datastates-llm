from setuptools import setup
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---- Install CPP/CUDA and expose Python bindings
name="datastates_src"
src_path_rel = "./"
src_path=pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/")

setup(
    name=name,
    version="0.0.1",
    author="ANL",
    ext_modules=[
        CUDAExtension(
            name=name,
            sources=[f'{src_path_rel}/pool/mem_pool.cpp', 
                    f'{src_path_rel}/tiers/gpu_tier.cpp', 
                    f'{src_path_rel}/tiers/host_tier.cpp', 
                    f'{src_path_rel}/engine.cpp', 
                    f'{src_path_rel}/py_datastates_llm.cpp'],
            include_dirs=[f"{src_path}/common", f"{src_path}/pool", f"{src_path}/tiers", f"{src_path}"],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2'] }
        )        
    ],
    install_requires=["pybind11", "torch"],
    cmdclass={
        'build_ext': BuildExtension
    })

# ---- Install actual DataStates APIs which will be invoked by deepspeed.
