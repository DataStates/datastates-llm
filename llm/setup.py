from setuptools import setup, find_packages

setup(
    name="datastates",
    version="0.0.1",
    author="ANL",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    description="Datastates-LLM checkpointing engine",
    install_requires=["nanobind", "torch"],
)