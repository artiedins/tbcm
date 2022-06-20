from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tbcm",
    version="0.3",
    description="Ten Bit Color Maps",
    author="Artie Dins",
    url="https://github.com/artiedins/tbcm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["tbcm"],
    install_requires=["matplotlib"],
)
