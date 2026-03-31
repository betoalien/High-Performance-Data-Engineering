from setuptools import setup, find_packages

setup(
    name="hyperframe",
    version="0.1.0",
    packages=find_packages(),
    package_data={"hyperframe": ["libs/**/*"]},
    python_requires=">=3.10",
    description="High-performance Rust-backed DataFrame for Python",
    long_description=(
        "HyperFrame is built progressively throughout the "
        "'High-Performance Data Engineering: Integrating Python and Rust' course."
    ),
)
