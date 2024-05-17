from setuptools import setup
import subprocess

subprocess.check_call(['pip', 'install', '-v', './datastates_src'])

name="datastates"
setup(
    name=name,
    version="0.0.1",
    author="ANL",
    include_package_data=True,
    install_requires=["torch"],
    tests_require=['pytest'],
    packages=['datastates']
)