import setuptools

CORTEX_MODEL_SERVER_VERSION="master"

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
        nucleus-builder=main:main
    """,
    project_urls={
        "Bug Reports": "https://github.com/cortexlabs/nucleus/issues",
        "Chat with us": "https://community.cortex.dev/",
        "Documentation": "https://github.com/cortexlabs/nucleus",
        "Source Code": "https://github.com/cortexlabs/nucleus",
    },
)
