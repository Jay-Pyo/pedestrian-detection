# ICV21-FinalProejct
## _Pedestrian Detection Algorithm Using Various Methods_

[![N|Solid](https://www.snu.ac.kr/webdata/uploads/kor/image/2019/12/about-symbold-logo-love_lg.png)](https://www.snu.ac.kr/)

>This project deals with pedestrian detection algorithms using [INRIAPerson Dataset](http://pascal.inrialpes.fr/data/human/). It is written with the intention of learning Non-Learning Computer vision.

## Reference Paper

- Navneet Dalal and Bill Triggs, Histograms of Oriented Gradients for Human Detection, CVPR 2005
- Paul Viola and Michael Jones, Rapid Object Detection using a Boosted Cascade of Simple Features, CVPR 2001
- Qiang Zhu et al., Fast Human Detection Using a Cascade of Histograms of Oriented Gradients, CVPR2006
- Y. Mu, S. Yan, Y. Liu, T. Huang, B. Zhou, Discriminative local binary patterns for human detection in personal album, CVPR 2008

## Features

- Pedestrian Detection Algorithm Using HOG & SVM
- Pedestrian Detection Algorithm Using Cascade
- Pedestrian Detection Algorithm Using LBP & SVM

## Pipeline
> How to run

Please install and untar [INRIAPerson Dataset](http://pascal.inrialpes.fr/data/human/)  in the same path as this file.
For generating(and augmentation) data...

```sh
python3 ./gen_data.py
```

For run **method I** (HOG & SVM)...

```sh
python3 ./train1.py
# wait until the training is over.
python3 ./test1.py
```

For run **method II** (Cascade)...

```sh
python3 ./train2.py
# wait until the training is over.
python3 ./test2.py
```

For run **method III** (LBP & SVM)...

```sh
python3 ./train3.py
# wait until the training is over.
python3 ./test3.py
```

## Environment

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Name | Version |
| ------ | ------ |
| Python3 | 3.6.9 |
| GCC | 8.3.0 on linux |
