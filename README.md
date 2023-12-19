# OsmLocator
A tool to localize overlapping scatter marks. This work formulates the scatter marks localization as an optimization problem on cluster-based re-visualization.

For a binary image of a scatter image, OsmLocator can locate the marks in the binary image. The following two images show the mark localization results of two scatter images in which black dots mark out the located position of scatter marks.

![Localization result of a scatter plot.](/images/scatter_plot.png "Localization result of a scatter plot.")
![Localization result of a scatter map.](/images/scatter_map.png "Localization result of a scatter map.")

If you would like to use or know about this package, please refer to and consider citing the following paper: 
```
@article{qiu2023osmlocator,
  title={OsmLocator: locating overlapping scatter marks by simulated annealing on clustering-based re-visualization},
  author={Yuming Qiu, Aleksandra Pizurica, Qi Ming and Nicolas Nadisic},
  journal={arXiv preprint arXiv:2312.11146},
  url={https://arxiv.org/abs/2312.11146},
  year={2023}
}
```

# Installation and Dependencies
You should have opencv and numpy library installed.
You should have opencv and numpy library installed.
* numpy
* opencv_python

The package is on PyPI, so simply run `pip install osmlocator` to install it.
If you want the newest and freshest version, you can install it by executing:
```
pip install git+https://github.com/ccqym/OsmLocator.git
```

# Dataset
We publish a new dataset SML2023 for overlapping scatter marks localization along with this work. You can find all images and corresponding annotations in the `datasets/sml_dataset2023` dir.
```shell
git clone https://github.com/ccqym/OsmLocator
cd OsmLocator/datasets/sml_dataset2023
ls
```

# Usage
You can use this package by two ways: library or CLI

## OsmLocator 
```python
# Firstly create a new OsmLocator instance.
locator = OsmLocator(settings)
# Then use the method locate to obtain the marks' location.
markLoctions = locator.locate(binImg)
```

### Examples
Let us take a case from SML2023 dataset to show how to use.
```python
import cv2
import numpy as np
import osmlocator as ol

settings = ol.getDefaultSetting()
locator = ol.OsmLocator(settings)

img = cv2.imread('cli_tool/imgs/sml_00001.png', cv2.IMREAD_GRAYSCALE)
if img is None: exit()
A_Threshold_Value = 200
binImg = (img<A_Threshold_Value).astype(np.uint8)*255 #The gray image can be converted into a binary image by OTSU algorithm. 
markLoctions = locator.locate(binImg)
print(markLoctions)
```

## evaluator
The data format is [{'x':p1_x, 'y':p1_y},...,{'x':pn_x, 'y':pn_y}]. Following the above example, the located `markLoctions` can be evaluated like the following codes.
```python
import json
import osmlocator as ol

jcont = json.load(open('cli_tool/imgs/sml_00001.json'))
gtMarkLocs = jcont['marks_location']
score = ol.evaluateXY(gtMarkLocs, markLoctions, img.shape)
print(score)
```
Evaluation results are a dict like
```python
{'ass_score': {
    '1': 0.9020016023684265, 
    '5': 0.9084003204736852, 
    '10': 0.9092001602368427
}}
```
the key '1', '5' and '10' are the $\lambda$ parameter of ACB metric.

## CLI
We have wrapped the functions of library as a CLI tool, so you can use the function by the following command.

The usage of the wrapper is as follow:
```shell
git clone https://github.com/ccqym/OsmLocator
cd OsmLocator/cli_tool/
# use '-H' to get usage information
python main.py -h
# you can run a shell script to run all examples.
./run_examples.sh
```

## License
Copyright (c) Cigit, Cas, China, and Ugent, Belgium. All rights reserved.
Licensed under the [MPL-2.0](LICENSE.txt) license.
