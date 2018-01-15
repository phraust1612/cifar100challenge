# cifar100challenge

This repository is to practice CNN with [cifar](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.  
There are 3 trainable networks in this repo.  
* [Resnet-152](https://arxiv.org/abs/1512.03385)
* [Lenet](http://yann.lecun.com/exdb/lenet/)
* Linear classifier

## Run

```shell
python3 train.py [Resnet or Lenet] --dataset [cifar10 or cifar 100] [-i]
```
where -i option is to initialize the network for the first time.  
You need to download Resnet caffemodel file to initialize resnet architecture.  
You can download it from [here](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777) and locate it in cnn_resnet/.  

## Components

* train.py : main py file to run training
* cifar.py : python3 API to load cifar dataset as a small batch
* net.py : abstract object of the network
* cnn_resnet/ : directory of Resnet-152, containining net source and parameters
* cnn_lenet/ : directory of Lenet, contatining net source and parameters
