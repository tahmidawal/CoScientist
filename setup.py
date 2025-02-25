from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="coscientist",
    version="0.1.0",
    description="AI Co-Scientist: Autonomous research idea generation and refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/coscientist",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "huggingface": ["huggingface-hub>=0.12.0"],
        "praison": ["praisonai>=0.1.0"],
        "gemini": ["google-generativeai>=0.3.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "huggingface-hub>=0.12.0",
            "praisonai>=0.1.0",
            "google-generativeai>=0.3.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "pylint>=2.8.0",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "coscientist=coscientist.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 