from distutils.core import setup

from setuptools import find_packages

setup(
        name="wake-chess",
        packages=find_packages(),
        version="0.0.1",
        license="Apache",
        description="A chess engine",
        author="Wes Doyle",
        url="https://github.com/wesdoyle",
        keywords=["chess", "chess-programming", "chess-engine", "bitboard", "game", "numpy"],
        )

