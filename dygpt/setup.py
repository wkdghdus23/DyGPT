from setuptools import setup, find_packages

setup(
    name='dygpt',
    version='1.0.0',
    author='Ho Yeon Jang',
    author_email='wkdghdus23@gmail.com',
    description='A unified GPT training package for GPT tutorials',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wkdghdus23/DyGPT',
    packages=find_packages(),
    install_requires=[
        'torch>=2.4.1',
        'transformers>=4.44.2',
        'tqdm>=4.66.5',
        'pandas>=2.0.3',
        'numpy>=1.24.3',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Framework :: PyTorch',
        'Author :: Ho Yeon Jang, Sogang University, Artificial Intelligence & Energy Materials Group'
    ],
    entry_points={
        'console_scripts': [
            'dygpt=dygpt.main:main'
        ],
    },
)


