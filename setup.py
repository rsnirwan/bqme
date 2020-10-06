import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bqme",
    version="0.1.0",
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
            "pytest-cov==2.10.1",
            "sphinx==3.2.1",
            "sphinx_rtd_theme==0.5.0",
            "recommonmark==0.6.0",
        ],
    },
    package_data={
        'bqme':['stan_code_template.stan',],
    },
    include_package_data=True,
)
