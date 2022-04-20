# ARRM

![LICENSE](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.7-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.1.0-%237732a8)

Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
1. activate Python environment
2. git clone https://github.com/NVIDIA/apex.git
3. cd apex
4. git checkout 96b017a
5. python setup.py install --cuda_ext --cpp_ext
6. cd ..
7. python setup.py build develop

## Prepare Pascal VOC datasets
First, you need to download the VOC datasets [here](https://drive.google.com/file/d/14muqZUdbpnYQ_30ZpAP9KqrVVHSkJOhU/view?usp=sharing).
Then, put "datasets" into this repository. The "datasets" contains the original VOC2007/2012 datasets and correspondiing class split. The "datasets" is shown below:

```bash
datasets/voc/
            ├──VOC2007
                  ├── Annotations
                  ├── ImageSets
                  ├── JPEGImages
                  ├── Crops
                  ├── Crops_standard-1shot
                  ├── Crops_standard-2shot
                  ├── Crops_standard-3shot
                  ├── Crops_standard-5shot
                  ├── Crops_standard-10shot
            ├──VOC2012
                  ├── Annotations
                  ├── ImageSets
                  ├── JPEGImages
                  ├── Crops
                  ├── Crops_standard-1shot
                  ├── Crops_standard-2shot
                  ├── Crops_standard-3shot
                  ├── Crops_standard-5shot
                  ├── Crops_standard-10shot
```

## Training
1. Run the following for base training and novel training on Pascal VOC splits-1.

```bash
bash tools/fewshot_exp/train_voc_all.sh 
```


