import setuptools

with open("requirements.txt") as fp:
    install_requires = fp.read()

setuptools.setup(
    name="cortex-model-server-builder",
    version="0.1.0",
    author="Robert Lucian Chiriac",
    author_email="robert@cortexlabs.com",
    description="A Python CLI tool to build model-server dockerfiles to be used together with the Cortex platform",
    url="https://github.com/cortexlabs/cortex-templates",
    py_modules=["main"],
    install_requires=install_requires,
    classifiers=[
        # TODO add the right classifiers
    ],
    include_package_data=True,
    package_data={"": ["templates/*"]},
    entry_points="""
        [console_scripts]
        cortex-model-server-builder=main:main
        cmsb=main:main
    """,
)
