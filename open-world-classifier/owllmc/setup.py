from setuptools import setup, find_packages

setup(
    name="owllmc",
    version="0.1.0",
    description="Open World LLM-based Classification (Generalist + Specialist)",
    author="Ricardo Marcacini, Daniel Zitei, Kenzo",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "openai",
        "tqdm",
        "langchain_community"
    ],
)
