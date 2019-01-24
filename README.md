# Semantic Segmentation
> Implementation of several Semantic Segmentation architectures for common segmentation tasks.


[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

This respository consists of the code for Semantic segmantation using several Deep Learning Architectures.


## Features

- [x] If space complexity and time complexity is taken into criteria, ENET is a viable option. It can be deployed even in mobile devices.
- [x] If the time and space complexity can be relaxed, UNET and Modified VGG net performs better than their other variant ENET.
- [x] The training code is very much scalable towards any new architecture.
- [x] All changes made in the config file will effect in the training process so that the training logic can be without hassle.
- [x] The training configuartion are easily tunable through the config file provided.

## Requirements

- The training module has been built using Pycharm 2018.1.4.
- The System requirement’s are 2.7 GHz Intel Core i5 with atleast 8 GB of RAM.

## Installation

#### OpenCV
You can use [Anaconda](https://conda.io/) to install `opencv` with the following command line.:

```
conda install -c conda-forge opencv
```

#### Image Augmentation
You can use [PIP](https://pypi.org/project/pip/) to install the module `imgaug` with the following command line.:

```
pip install imgaug
```

#### Tensorflow
You can use [PIP](https://pypi.org/project/pip/) to install `tensorflow` with the following command line or please go through their official installation [guideline](https://www.tensorflow.org/install/pip)
```
pip install tensorflow
```


#### Keras
You can use [PIP](https://pypi.org/project/pip/) to install `keras` with the following command line or please go through their official installation [guideline](https://keras.io/#installation)

```
pip install keras
```

## Usage example

Run the following script to dispatch the trainer.


```
python3 train.py  --conf=./config.json
```

## Contribute

Don't feel shy to drop a star, if you find this repo useful. I would love for you to contribute to **KITT-Road Segmentation**, check the ``LICENSE`` file for more info.

## Meta

Stanly Moses – [@Linkedin](https://in.linkedin.com/in/stanlymoses) – stanlyhardy@yahoo.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/StanlyHardy/Semantic-Segmentation](https://github.com/StanlyHardy/)

