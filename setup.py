from setuptools import setup, find_packages
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Define CPP/CUDA extensions for the checkpointing engine.
ckpt_engine_path = "datastates/ckpt/src/"
abs_ckpt_engine_path=pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/{ckpt_engine_path}")
cmake_flags = [
    "FMT_HEADER_ONLY=1",
    "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    "SPDLOG_FMT_EXTERNAL",
    "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA",
    "THRUST_DISABLE_ABI_NAMESPACE",
    "THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP",
    "THRUST_IGNORE_ABI_NAMESPACE_ERROR",
]
rmm_flags = [f'-D{flag}' for flag in cmake_flags]
extensions = [
    CUDAExtension(
        name=f"datastates.ckpt.src",
        sources=[f'{ckpt_engine_path}/tiers/gpu_tier.cpp', 
                    f'{ckpt_engine_path}/tiers/host_tier.cpp', 
                    f'{ckpt_engine_path}/engine.cpp', 
                    f'{ckpt_engine_path}/py_datastates_llm.cpp'],
        include_dirs=[f"{abs_ckpt_engine_path}/common", 
                      f"{abs_ckpt_engine_path}/pool", 
                      f"{abs_ckpt_engine_path}/tiers", 
                      f"{abs_ckpt_engine_path}",
                      '/soft/compilers/cuda/cuda-12.2.2/include',
                      '/home/amaurya/.conda/envs/dspeed_env/include/rapids',
                      '/home/amaurya/.conda/envs/dspeed_env/include/rapids/libcudacxx',
                      '/home/amaurya/.conda/envs/dspeed_env/include'                      
                      ],
        extra_compile_args={'cxx': ['-g', '-fvisibility=hidden', '-std=c++17', '-Wall', '-O0','-Wno-reorder'] + rmm_flags, 
                            'nvcc': ['-Wall', '-O0', '-std=c++17', '--expt-extended-lambda', '-arch=sm_80'] + rmm_flags }, 
        libraries=['cudart', 'cuda', 'fmt', 'spdlog'],
        # Need fvisibility for smaller binaries: 
        # https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-create-smaller-binaries
    )    
]

# ---- Install CPP/CUDA checkpointing engine and expose Python bindings
setup(
    name="datastates",
    version="0.0.1",
    author="DataStates Team (Argonne National Laboratory)",
    packages=['datastates', 'datastates.utils', 'datastates.ckpt', 'datastates.llm', 'datastates.llm.deepspeed'],
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