from setuptools import find_packages, setup
import os
import subprocess
import sys


if __name__ == "__main__":
    fp = open("xdit_comfyui_private/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="xdit-comfyui-private",
        author="xDiT Team",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=[],
        url="https://github.com/xdit-project/xdit-comfyui-private.",
        description="xDiT: A Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters",
        # long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
