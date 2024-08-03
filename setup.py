from setuptools import setup, find_packages

setup(
    name='xgenius',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rich',
        'paramiko',
        'scp',
    ],
    entry_points={
        'console_scripts': [
            'xgenius=xgenius.cli:main',
            'xgenius-setup-clusters=scripts.setup_clusters:main',
            'xgenius-setup-runs=scripts.generate_run_config:main',
            'xgenius-build-image=scripts.build_image:main',
        ],
    },
    author='Roger Creus Castanyer',
    author_email='creus99@gmail.com',
    description='A tool for managing cluster jobs and configurations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/roger-creus/xgenius',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)