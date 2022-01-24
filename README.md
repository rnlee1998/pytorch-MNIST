# pytorch--MNIST

## Learning rate set
|model| lr|
|--|--|
|MLP|0.9|
|LeNet|0.9|
|AlexNet |0.01|

## train
```python
python train.py args_train.txt
```

## test
```python
python test.py args_test.txt
```

## test a demo
```python
python demo.py
```
MLP & LeNet input: 28 * 28

AlexNet input: 224 * 224

|model| acc_g|
|--|--|
|MLP|0.36875|
|LeNet|0.45625|
|AlexNet |0.56875|