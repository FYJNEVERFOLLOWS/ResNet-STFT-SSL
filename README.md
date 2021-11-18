# CNN-STFT-SSL
CNN-STFT Model for Sound Source Localization

Unofficial PyTorch implementation of He's: [Neural Network Adaptation and Data Augmentation for Multi-Speaker Direction-of-Arrival Estimation](https://ieeexplore.ieee.org/document/9357962)

![Overview of CNN-STFT](https://tva1.sinaimg.cn/large/008i3skNly1gvsy23r1t5j30fp0o6abz.jpg)

Dependency
----------

* `PyTorch <https://pytorch.org/>`
* `apkit <https://github.com/hwp/apkit>`_ (version 0.2)


Data
----

We use the `SSLR dataset <https://www.idiap.ch/dataset/sslr>`_ for the experiments.


## Usage
1. Run ./qsub/gen_data_frame_level.sh to extract features, then write them and the corresponding label into pickle file
2. Run ./qsub/train_with_CNN-STFT.sh and that's it. (If you don't want to select the two-stage training strategy, you may Run ./qsub/train_with_CNN-STFT-wo2stage.sh and that's it.)
