# Copyright 2021 Cortex Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

CORTEX_MODEL_SERVER_VERSION = "master"

with open("requirements.txt") as fp:
    install_requires = fp.read()

setuptools.setup(
    name="nucleus",
    version=CORTEX_MODEL_SERVER_VERSION,
    description="CLI tool to generating the dockerfiles of a nucleus model server; compatible with a Cortex cluster",
    author="Robert Lucian Chiriac",
    author_email="hello@cortex.dev",
    license="Apache License 2.0",
    url="https://github.com/cortexlabs/nucleus",
    py_modules=["main"],
    install_requires=install_requires,
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
    ],
    include_package_data=True,
    package_data={"": ["templates/*"]},
    entry_points="""
        [console_scripts]
        nucleus=main:cli
    """,
    project_urls={
        "Bug Reports": "https://github.com/cortexlabs/nucleus/issues",
        "Chat with us": "https://community.cortex.dev/",
        "Documentation": "https://github.com/cortexlabs/nucleus",
        "Source Code": "https://github.com/cortexlabs/nucleus",
    },
)
