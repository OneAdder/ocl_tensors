from setuptools import setup

setup(
    name='ocl_tensors',
    version=0.1,
    description='A tool for tensor operations with `OpenCL`',
    author='Michael Voronov',
    license='GPLv3',
    packages=['ocl_tensors'],
    python_requires='>=3.6',
    package_data={'ocl_tensors': ['kernel.cl']},
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy>=1.16.4', 'pyopencl>=2018.2.2'],
    extras_require={'test': ['pytest>=3.6']},
)
