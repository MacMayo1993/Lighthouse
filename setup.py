from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='pelt-lighthouse-antipodes',
    version='0.1.0',
    author='Lighthouse Research Team',
    description='Computationally efficient seam detection pipeline for time-series via PELT-Lighthouse-Antipodes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/Lighthouse',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'wfdb>=4.0.0',
        'ruptures>=1.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'pelt-lighthouse=pelt_lighthouse:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
