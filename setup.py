import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bqme",
    version="0.0.1",
    author="Rajbir Singh Nirwan",
    author_email="rajbir.nirwan@gmail.com",
    description="Bayesian Quanile Matching Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RSNirwan/BQME",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pystan>=2.19',
    ],
    extras_require={
        "dev":[
            "pytest>=6.0",
        ],
    },
)
