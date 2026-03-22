from pathlib import Path
import setuptools


def _read_requirements(path):
    reqs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


reqs = _read_requirements("requirements.txt") if Path("requirements.txt").is_file() else []
long_description = Path("README.md").read_text(encoding="utf-8")

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
