from setuptools import setup


setup(
    name='deep_cstrd',
    version='1.0.0',
    description='Cross section tree ring detection method over RGB images based on U-Net',
    url='https://github.com/hmarichal93/deepcstrd',
    author='Henry Marichal',
    author_email='hmarichal93@gmail.com',
    license='MIT',
    packages=['deep_cstrd'],
    classifiers=[
        'Development Status :: 1 - Review',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.12',
    ],
)