from setuptools import setup, find_packages

setup(
    name="issue_analyzer",
    version="0.1.0",
    description="LLM-based issue classification and analysis tool",
    author="LLM4Gov & Websensors Initiative",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "langchain",
        "openai",
        "pydantic",
        "pyyaml",
        "structlog",
        "pandas",
        "langchain_community",
        "langchain-openai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "analyze-issue=cli:main"
        ]
    }
)
