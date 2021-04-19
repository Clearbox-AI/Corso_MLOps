from setuptools import setup, find_packages


with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="production_model",
    version="0.0.1",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gillus/MLops_part1",
    install_requires=requirements,
    packages='',
    python_requires='>=3.6.2',
)
