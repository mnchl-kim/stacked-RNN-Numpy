# Character-level Language Model with stacked RNN (ONLY Numpy)

I implemented class module for stacked Recurrent Neural Networks with ONLY Numpy package.  

The following parameters can be selected in the RNN class.
 - input size
 - output size
 - hidden unit size
 - time length
 - depth size
 - batch size
 - dropout rate
 - learning rate

<br>

Additional information for RNN class,
 - Weight initialization : Xavier initialization  
 - Weight update optimizer : Adagrad, RMSProp  
 - Dropout  

<br>

To test the stacked RNN model, I used text data in online.  
It is for sequence generation similar with character-level language model.  
You can find out the text data used for training in data directory.

<br>

### Install environments
See `requirements.txt`

Select one of install methods below  

* Install all required packages with only one command line  
$ pip install --upgrade -r requirements.txt

* Install required packages individually  
numpy == 1.17.4  
matplotlib == 3.1.1

If `*.py` file doesn't run after installing required packages, check 'My working environment' in `requirements.txt`

<br>

### Source code

* `utils.py` : Includes several necessary function for running the other source code
* `model.py` : class RNN (stacked Recurrent Neural Networks) with ONLY Numpy
* `train.py` : Train data with RNN class
* `test.py` : Test file for live demo

If you want to check the training process, run `train.py`  
If you want to check the final result, run `test.py`. <br>
  
* `data` : Text data that I used for training the RNN model
* `figure` : Loss, Iteration graph
* `models` : Pre-implemented short codes for RNN, LSTM
* `ppt` : Presentation file with details for this project
* `result` : Results file stored by RNN object format(.pickle)

<br>

### Reference

https://github.com/janivanecky/Numpy-RNNs  
https://gist.github.com/karpathy/d4dee566867f8291f086

<br>

### License

MIT License
