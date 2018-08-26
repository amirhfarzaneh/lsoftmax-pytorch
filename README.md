# The Pytorch Implementation of L-Softmax
[//]: # (Image References)

[vis]: images/2d_vis.png "Training Loss"

this repository contains a new, clean and enhanced pytorch implementation of L-Softmax proposed in the following paper:

**Large-Margin Softmax Loss for Con volutional Neural Networks** By Weiyang Liu, Yandong Wen, Zhiding Yu, Meng Yang [[pdf in arxiv](https://arxiv.org/pdf/1612.02295.pdf)] [[original CAFFE code by authors](https://github.com/wy1iu/LargeMargin_Softmax_Loss)]

L-Softmax proposes a modified softmax classification method to increase the inter-class separability and intra-class compactness.

this re-implementation is based on the earlier pytorch implementation [here](https://github.com/jihunchoi/lsoftmax-pytorch) by [jihunchoi](https://github.com/jihunchoi) and borrowing some ideas from its TensorFlow implementation [here](https://github.com/auroua/L_Softmax_TensorFlow) by [auroua](https://github.com/auroua). Generally the improvements are as follows:
- [x] Now features visualization as depicted in the original paper using the `vis` argument in the code.
- [x] Cleaner and more readable code
- [x] More comments in `lsoftmax.py` file for future readers
- [x] Variable names are now in better correspondence with the original paper
- [x] Using the updated PyTorch 0.4.1 syntax and API
- [x] Two models to produce visualization in paper's fig 2 and the original MNIST model is provided
- [x] The lambda (`beta` variable in code) optimization missing in the earlier PyTorch code has been added (refer to section 5.1 in the original paper)
- [x] The numerical error of `torch.acos` has been addressed
- [x] Provided training logs in the [Logs](Logs/) folder
- [x] Some other minor performance improvements

### Version compatibility

This code has been tested in Ubuntu 18.04 LTS using PyCharm IDE and a NVIDIA 1080Ti GPU. Here is a list of libraries and their corresponding versions:

```
pytorch = 0.4.1
torchvision = 0.2.1
matplotlib = 2.2.2
numpy = 1.14.3
scipy = 1.1.0
```
### Network parameters
- batch_size = 256
- max epochs = 100
- learning rate = 0.1 (0.01 at epoch 50 and 0.001 at epoch 65)
- SGD with momentum = 0.9
- weight_decay = 0.0005

### Results

Here are the test set visualization results of training the MNIST for different margins:
![alt text][vis]
* this plot has been generated using the smaller network proposed in the paper for visualization purposes only with batch size = 64, constant learning rate = 0.01 for 10 epochs, and no weight decay regularization.

And here is the tabulated results of training MNIST with the proposed network in the paper:

| margin | test accuracy | paper |
|--------|---------------|-------|
| m = 1  | 99.37%        | 99.60%|
| m = 2  | 99.60%        | 99.68%|
| m = 3  | 99.56%        | 99.69%|
| m = 4  | 99.61%        | 99.69%|
* the test accuracy values are the max test accuracy of running the code only once with the network parameters above!
