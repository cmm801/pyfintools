from setuptools import setup, find_packages

setup(
    name='pyfintools',
    version='0.1.0',
    author='Christopher Miller',
    author_email='cmm801@gmail.com',
    packages=find_packages(include=['pyfintoos', 'pyfintoos.*']), 
    #include_package_data=True,
    scripts=[],
    url='http://pypi.python.org/pypi/pyfintools/',
    license='MIT',
    description='A package for working with financial instruments and data.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'pandas',
        'setuptools-git',
    ],
)
