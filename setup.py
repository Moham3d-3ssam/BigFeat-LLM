from setuptools import setup

# NOTE: packages=['setup'] matches this project's actual folder name
# (BigFeat-LLM/setup/). Installing this with `pip install .` registers a
# top-level importable module literally named "setup" in whatever Python
# environment you install it into - not just inside this project. If you
# ever hit an import conflict with something else called "setup", rename
# the package folder (and update the imports inside bigfeat_base.py,
# llm_runner.py, and __init__.py from `setup.xxx` to the new name) before
# using this file.

setup(
    description=(
        "Automated feature engineering library, with an optional "
        "LLM-driven layer for initializing hyperparameters and "
        "feature-importance seeds from a dataset description."
    ),
    name="bigfeat-llm",
    version="0.2",
    # author = "Hassan Eldeeb, Mohamed Essam",
    # author_email = "firstName.lastName@ut.ee",
    license="MIT",
    keywords=[
        "feature engineering",
        "machine learning",
        "automl",
        "feature extraction",
        "feature selection",
        "llm",
    ],
    url="https://github.com/DataSystemsGroupUT/BigFeat",
    packages=["setup"],
    long_description=(
        "Automated feature engineering library, with an optional "
        "LLM-driven layer for initializing hyperparameters and "
        "feature-importance seeds from a dataset description."
    ),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "scipy",
        "requests",
    ],
    extras_require={
        # pip install .[gemini]
        "gemini": ["google-genai"],
    },
)