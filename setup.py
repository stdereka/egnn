from setuptools import setup, find_namespace_packages


setup(
    version="0.1",
    name="egnn",
    url="https://github.com/stdereka/egnn",
    author="Stanislav Dereka",
    author_email="st.dereka@gmail.com",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==1.13.0",
        "pytorch_lightning==1.8.1",
        "numpy==1.23.4",
        "yaml"
    ]
)
