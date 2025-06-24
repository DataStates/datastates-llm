from setuptools import setup, Extension, find_packages
import pathlib
import glob
import os
import sys
import subprocess
from setuptools.command.build_ext import build_ext

# Discover the shared object built by CMake manually
# datastates_ckpt_so = glob.glob("datastates/ckpt/*.so") + glob.glob("datastates/ckpt/*.pyd")

# Custom build class to run CMake
class CMakeBuild(build_ext):
    def run(self):
        # Make sure CMake is installed
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is required to build this project.")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Path setup
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]

        build_args = ['--config', 'Release', '--', f'-j{os.cpu_count()}']

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake configure + build
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

# Dummy Extension that just points to the CMake root
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


setup(
    name="datastates",
    version="0.0.1",
    author="ANL",
    packages=find_packages(include=['datastates', 'datastates.*']),
    include_package_data=True,
    description="Datastates-LLM checkpointing engine",
    install_requires=["nanobind", "torch"],
    cmdclass={'build_ext': CMakeBuild},

)