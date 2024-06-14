from setuptools import setup, find_packages
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---- Install CPP/CUDA and expose Python bindings
# name="datastates.ckpt"
src_path_rel = "./src/"
src_path=pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/src/")

setup(
    name="datastates",
    version="0.0.1",
    author="ANL",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name=f"datastates.ckpt",
            sources=[f'{src_path_rel}/pool/mem_pool.cpp', 
                    f'{src_path_rel}/tiers/gpu_tier.cpp', 
                    f'{src_path_rel}/tiers/host_tier.cpp', 
                    f'{src_path_rel}/engine.cpp', 
                    f'{src_path_rel}/py_datastates_llm.cpp'],
            include_dirs=[f"{src_path}/common", f"{src_path}/pool", f"{src_path}/tiers", f"{src_path}"],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2'] },
            is_standalone=False,
            is_python_module=False,
            verbose=True
        )        
    ],
    install_requires=["pybind11", "torch"],
    cmdclass={
        'build_ext': BuildExtension
    })

# ---- Install actual DataStates APIs which will be invoked by deepspeed.
