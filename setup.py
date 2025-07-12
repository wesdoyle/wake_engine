from distutils.core import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="wakechess",
    packages=find_packages(),
    version="0.0.1",
    license="Apache",
    description="A bitboard-based chess engine",
    author="Wes Doyle",
    url="https://github.com/wesdoyle",
    keywords=[
        "chess",
        "chess-programming",
        "chess-engine",
        "bitboard",
        "game",
        "numpy",
    ],
    install_requires=requirements,
)
