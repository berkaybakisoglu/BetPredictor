from setuptools import setup, find_packages

setup(
    name="betpredictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "tqdm>=4.62.3",
        "aiohttp>=3.8.0",
        "pandas-ta>=0.3.14b0",
    ],
) 