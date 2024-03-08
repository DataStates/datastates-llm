from setuptools import setup
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---- Install CPP/CUDA and expose Python bindings
name="datastates_cpp"
src_path_rel = "./"
src_path=pathlib.Path(f"{pathlib.Path(__file__).parent.resolve()}/")

setup(
    name=name,
    version="0.0.1",
    author="ANL",
    ext_modules=[
        CUDAExtension(
            name=name,
            sources=[f'{src_path_rel}/host_cache.cpp', f'{src_path_rel}/ckpt_engine.cpp', f'{src_path_rel}/py_datastates_llm.cpp'],
            include_dirs=[src_path],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2'] }
        )        
    ],
    install_requires=["pybind11", "torch"],
    cmdclass={
        'build_ext': BuildExtension
    })

# ---- Install actual DataStates APIs which will be invoked by deepspeed.
