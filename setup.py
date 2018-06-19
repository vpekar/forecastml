from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6'
]

setup(
    name="forecastml",
    version="0.0.1",
    description="""Machine learning regression on time-series data.""",
    long_description="""A Python package for running experiments with machine learning regressors on time-series data.""",
    author="Viktor Pekar",
    author_email="v.pekar@gmail.com",
    url="https://github.com/vpekar/forecastml",
    license="MIT License",
    keywords="machine-learning regression time-series",
    classifiers=classifiers,
    packages=find_packages(),
    include_package_data = False,
    install_requires = [
      'numpy',
      'pandas',
      'statsmodels',
      'scikit-learn',
      'xgboost'
    ],
    tests_require = [
      'nose',
    ],
    py_modules=['forecastml'],
    zip_safe = False,
    entry_points = {
        'nose.plugins.0.10': [
            'yamltests = yamltests:YamlTests'
            ]
        },
)