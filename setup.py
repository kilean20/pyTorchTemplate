import os
from distutils.core import setup
import sys


setup(
    name = "pyTorchTemplate",
    version = "0.0.1",
    author = "Kilean Hwang",
    author_email = "kilean@lbl.gov",
    description = ("templates for pytorch apis"),
    license = "Lawrence Berkeley National Laboratory",
    keywords = "pyTorch", "template"
    url = "",
    packages=['pyTorchTemplate'],
    #package_data={'pImpactR': ['xmain']},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Utilities",
        "License :: Free for non-commercial use",
    ],
    zip_safe=False
)
