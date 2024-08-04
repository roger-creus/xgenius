from setuptools import setup, find_packages

setup(
    name='xgenius',
    version='0.1.8',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rich',
        'paramiko',
        'scp',
    ],
    package_data={
        'xgenius': ['sbatch_templates/*'],  # Include sbatch_templates in the package
    },
    entry_points={
        'console_scripts': [
            'xgenius=xgenius.cli:main',
            'xgenius-setup-clusters=xgenius.scripts.setup_clusters:main',
            'xgenius-setup-runs=xgenius.scripts.generate_run_config:main',
            'xgenius-build-image=xgenius.scripts.build_image:main',
            'xgenius-check-jobs=xgenius.scripts.check_jobs:main',
            'xgenius-cancel-jobs=xgenius.scripts.cancel_jobs:main',
            'xgenius-pull-results=xgenius.scripts.pull_results:main',
            'xgenius-batch-submit=xgenius.scripts.batch_submit:main',
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
