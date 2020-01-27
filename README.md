## Neural Network
The aim of this project is to implement a neural network with two hidden layers. There are four  parameters to configure the neural network:
* **c**: number of files in the dataset folder
* **w**: number of elements taken from each row as a window
* **k**:  number of neurons in output layer
* **stride**: the offset over the window w

The dataset link can be found [here](https://drive.google.com/open?id=1wuB1T3Z74ag258cF4DihLsoDeeiyNmYR).
## Description
The input layer has **c** x **w** neurons, the hidden layers have **c** neurons each and the output layer has **k** neurons. The two hidden layers and output layer are dense (fully connected) while for j = 1, the neuron, where for j = 1, the neuronIJ,
0 < i ≤ w, 0 < j ≤ c of the first input layer is connected to neuronJ of the first hidden layer. Moreover, the weights between the 2 hidden layers should belong to the range of [−1, 1] and the activation function must be sigmoid. In order to visualize this, here is an example for c=3, w=3 and k=6:

<p align="center">
  <img src="https://github.com/sammy9867/neural_network/blob/master/images/nn.png?raw=true" alt="Neural Network"/>
</p>

## Getting Started
In order to run our program, we need to open the command line and switch to the directory
containing the python file and enter the following command:
```
python neural_network.py "absolute_path/dataset_name" w stride epochs learning_rate
```
- **absolute_path/dataset_name**: contains the absolute path of the dataset where dataset_name is a folder containing train and test files (csv/arff) [Note: path should be within the quotations]
- **w**: the number of elements taken from each row as a window. (e.g.: 5)
- **stride**: the offset over the window w. (e.g.: 2)
- **epochs**: the number of times our neural network is trained on the given dataset. (e.g.: 100)
- **learning_rate**: initial learning rate for the neural network. (e.g.: 0.01)

## Technology used
I have used Python due to its simplicity. Apart from the standard inbuilt python modules, I have also used additional modules to code my neural network namely:
* **numpy**: For operations on large multi-dimensional arrays and matrices.
* **pandas**: To read CSV files.
* **scipy**: To read ARFF files.
* **sklearn.metrics**: To compute confusion matrix and overall accuracy.
* **glob**: To find all the pathnames matching a specific pattern

## Author
* **Samuel Menezes**
