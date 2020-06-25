IPSTERS proj - Main Experiments 
===============================

In this repository you can find most of the experiments I developed for the project "IPSTERS - IPSentinel Terrestrial Enhanced Recognition System", funded by "Fundação para a Ciência e a Tecnologia". For full info please follow [this link](https://joaopfonseca.github.io/projects/ipsters/).

Project Organization
------------

    .
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── applications
    │   ├── COSsim
    │   └── T29SNC
    ├── data/
    ├── docs/
    ├── models/
    ├── notebooks/
    ├── pipelines/
    ├── references/
    ├── reports/
    ├── requirements.txt
    ├── setup.py
    ├── src/
    │   ├── __init__.py
    │   ├── data/
    │   ├── experiment/
    │   │   ├── __init__.py
    │   │   └── utils.py
    │   ├── models
    │   │   ├── AutoEncoder.py
    │   │   ├── HybridSpectralNet.py
    │   │   ├── __init__.py
    │   │   ├── denoiser.py
    │   │   ├── recurrent.py
    │   │   └── resnet.py
    │   ├── preprocess
    │   │   ├── __init__.py
    │   │   ├── data_selection.py
    │   │   ├── feature_selection.py
    │   │   ├── readers.py
    │   │   ├── relieff.py
    │   │   └── utils.py
    │   └── reporting
    │       ├── __init__.py
    │       ├── reports.py
    │       └── visualize.py
    ├── test_environment.py
    ├── texput.log
    └── tox.ini

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
