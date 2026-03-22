import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="saccard",
    version="1.0",
    author="dr-r-kan",
    author_email="",
    description="Remote cardiac data extraction from video using rPPG methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dr-r-kan/saccard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_data={"": ['*.pth', '*.png', '*.hdf5']},
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=reqs,
)
