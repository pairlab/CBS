# Curriculum by Smoothing (NeurIPS 2020)

The official PyTorch implementation of Curriculum by Smoothing (NeurIPS 2020, Spotlight).

For any questions regarding the codebase, please send a message at: samarth.sinha@mail.utoronto.ca

***USAGE:***
Simply use the code by running:

`python3 main.py --dataset <DATASET> --alg <MODEL> --data <PATH_TO_DATA>`

For example, to train a ResNet on CIFAR10 and the data is saved in `./data/`, we can run:

`python3 main.py --dataset cifar10 --alg res --data ./data/`


For new expeiments it is important to tune the following hyperparameters:

`--std --std_factor --epoch`

***LINK***: https://arxiv.org/abs/2003.01367

This codebase has experiments for image classification and transfer learning.

If you use this codebase or find this repository helpful then please cite our paper:
```
@article{sinha2020curriculum,

  title={Curriculum By Smoothing},

  author={Sinha, Samarth and Garg, Animesh and Larochelle, Hugo},

  journal={Advances in Neural Information Processing Systems},

  volume={33},

  year={2020}
}
```
