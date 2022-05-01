# AutoRegressive Models for Time Series Forecasting

## Intro
Autoregressive models infer the time relation in a time series by studying the relation between the past data (also colled **lags**) and the data at a certain time **horizon** in the future


 *f(y<sub>t-1</sub>, ... ,y<sub>t-d</sub>) = y<sub>t+h</sub> .*

Here *f* denotes the learned relation which depends on the machine learning estimator used to train the autoregresive model.

## Installation
To use this functions simply donwload the file to your working directory. The required packages are `pandas` , `numpy` and `sklearn`.


## Examples
The file **test.py** contains examples on how to use this model. 

## Future extensions
Right now the model is thought for learning from one-dimensional time series data. 
In the future I would like to extend it in order to cover multidimensional time series or add tools to generate features of the data.
