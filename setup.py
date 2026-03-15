from setuptools import setup, find_packages

setup(
    name='xgenius',
    version='2.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rich',
        'paramiko',
        'scp',
        'tomli_w',
    ],
    package_data={
        'xgenius': ['sbatch_templates/*'],
    },
    entry_points={
        'console_scripts': [
            'xgenius=xgenius.cli:main',
        ],
    },
    author='Roger Creus Castanyer',
    author_email='creus99@gmail.com',
    description='LLM-oriented autonomous research platform for SLURM clusters',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/roger-creus/xgenius',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
