from setuptools import setup, find_packages
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Define CPP/CUDA extensions for the checkpointing engine.
ckpt_engine_path = "datastates/ckpt/src/"
abs_ckpt_engine_path=pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/{ckpt_engine_path}")
extensions = [
    CUDAExtension(
        name=f"datastates.ckpt.src",
        sources=[f'{ckpt_engine_path}/pool/mem_pool.cpp', 
                    f'{ckpt_engine_path}/tiers/gpu_tier.cpp', 
                    f'{ckpt_engine_path}/tiers/host_tier.cpp', 
                    f'{ckpt_engine_path}/engine.cpp', 
                    f'{ckpt_engine_path}/py_datastates_llm.cpp'],
        include_dirs=[f"{abs_ckpt_engine_path}/common", 
                      f"{abs_ckpt_engine_path}/pool", 
                      f"{abs_ckpt_engine_path}/tiers", 
                      f"{abs_ckpt_engine_path}"],
        extra_compile_args={'cxx': ['-g', '-fvisibility=hidden'], 'nvcc': ['-O2'] } 
        # Need fvisibility for smaller binaries: 
        # https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-create-smaller-binaries
    )    
]

# ---- Install CPP/CUDA checkpointing engine and expose Python bindings
setup(
    name="datastates",
    version="0.0.1",
    author="ANL",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    verbose=True,
    description="Datastates-LLM checkpointing engine",
    long_description=open("README.md").read() if pathlib.Path("README.md").exists() else "",
    install_requires=["pybind11", "torch"],
    cmdclass={
        'build_ext': BuildExtension
    }    
)