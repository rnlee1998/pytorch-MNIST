# pytorch--MNIST



## train
```python
python train.py args_train.txt
```
Learning rate set
|Model| lr|
|--|--|
|MLP|0.9|
|LeNet|0.9|
|AlexNet |0.01|
|VGG|0.05|

## test
```python
python test.py args_test.txt
```

## test a demo
```python
python demo.py
```
MLP & LeNet input: 28 * 28

AlexNet & VGG input: 224 * 224

|model| acc_g|
|--|--|
|MLP|0.36875|
|LeNet|0.45625|
|AlexNet |0.59375|
|VGG|0.65|