# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:39:22 2021

@author: nicopaglia
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import time

from itertools import product as cart_prod

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit as TSSplit


# ============================================================================= 
# AutoReg class
# ============================================================================= 

class AutoReg:
    """
    Instances:
        - estimator: learning estimator with methods:
                        - .fit 
                        - .predict 
                        - .set_params
        - d: size of the previous lag to consider in the model
        - H: time horizon to predict in the future
        - series: series to apply the methods to
        - fitted_indices: indices of the series used to fit the model
        - fitted_series: series used for training (=series[fitted_indices]))
        - history_estimators: list of all the fitted estimators
        - history_t: indices of re-training
        - history_d: list of all the lag length considered
    
    Methods:
        - fit: used to fit the learning methods using a slice of 
               the timese series
        - predict_indices: predict some indices in the time series
        - predict_next_horizon: predict Hth future values of the series
        - prediction_error: compute the prediction error on some indices 
                            of the series
        - CV_params: parameters selection via cross validation 
                     with Time series splitting
        - CV_fit: fit a slice of time series with optimal CV params. 
        
    """
    
    
# =============================================================================    
    def __init__(self, 
                 estimator=LinearRegression(), 
                 d=1, 
                 H=1,
                 series=None, 
                 fitted_indices=None, 
                 fitted_series=None,
                 cv=None, 
                 history_estimators= [], 
                 history_t= [],
                 history_d= [], 
                 history_cv=[]):
        """Initializer"""
        self.estimator=estimator
        self.d = d
        self.H = H
        self.series=series
        self.fitted_indices=fitted_indices
        self.fitted_series=fitted_series
        self.cv= cv
        self.history_estimators=history_estimators
        self.history_t=history_t
        self.history_d=history_d
        self.history_cv=history_cv

    
    def fit(self,series=None,
            idx_start=0, 
            idx_end=-1, 
            add2history=False):
        """
        Fit the estimator using a slice of a time series.
        """
        if series is None:
            series=self.series.copy()
        df, X, Y = create_dataset(series,idx_start, idx_end, d=self.d, H=self.H)
        self.estimator.fit(X,Y)
        # New attributes on the data used to fit
        self.series=series
        self.fitted_indices=np.arange(idx_start,idx_end,1)
        self.fitted_series=series[self.fitted_indices]
        if add2history:
            self.update_history()
        return self

    
    def predict_indices(self,
                        idx_start,
                        idx_end=None, 
                        series=None):
        """
        Predict a list of indices in the time series.
        """
        if series is None:
            series = self.series.copy()
        if idx_end is None:
            idx_end=idx_start+1
        X = create_inputs(series,idx_start-self.d-self.H+1, idx_end-self.H, d=self.d)
        Y_pred = self.estimator.predict(X)
        pred=pd.Series(index=series.index[idx_start:idx_end],
                       data=Y_pred)
        return pred

    
    def predict_next_horizon(self, series=None):
        """
        Predict the next values of the series.
        """
        if series is None:
            series=self.series.copy()  
        
        idx_start = len(series)-self.d
        idx_end   = len(series)
        x = series.values[idx_start:idx_end]
        print('Input: ', x)
        pred =(self.estimator.predict(x.reshape(1, -1))[0])
        print(f'Prediction H={self.H}:', pred)
        return pred

    
    
    def predict_next_steps(self, steps=1, series=None):
        """
        Predict the next values of the series.
        """
        if series is None:
            series=self.series.copy()  
        
        dt_diff = series.index[-1]-series.index[-2]
        for i in range(steps): 
            last_dt = series.index[-1]
            idx_start = len(series)-self.d-self.H+1  
            idx_end   = len(series)-self.H+1
            x = series.values[idx_start:idx_end]
            print('New input: ', x)
            pred =(self.estimator.predict(x.reshape(1, -1))[0])
            print('New prediction:', pred)
            series.loc[last_dt+dt_diff] = pred
            
        return series[-steps:]

    
    def prediction_error(self,
                         idx_start, 
                         idx_end=None, 
                         series=None, 
                         metric=mean_absolute_error,
                         pointwise=False):
        """
        Score the learning method by comparing the prediction against
        the true values on some indices with a specified metric.   
        """
        if series is None:
            series=self.series.copy()
        if idx_end is None:
            idx_end=idx_start
        pred = self.predict_indices(idx_start,idx_end,series)
        true=series.values[idx_start:idx_end]
        if pointwise is False:
            return metric(true,pred) 
        else: 
            err=np.zeros(len(true))
            for i in range(len(true)):
                err[i] = metric([true[i]],[pred[i]])
            return err


    def KNN_CV(self,idx_start,idx_end,
               kmin=1, kmax=None, knum=10,
               ts_split = True,
               n_splits=5, 
               series=None, add2history=True, get_info=False):
        """
        K nearest neighbors with cross validation over k
        """
        if series is None:
            series=self.series.copy() 
        # Create the training set
        df, X, Y = create_dataset(series,idx_start, idx_end, d=self.d, H=self.H)
        
        if (kmax is None) or (kmax>= X.shape[0]):
            kmax=np.ceil(X.shape[0]/2)
            
        # Initialize cv list
        cv = []
        #yvar = np.var(Y) #variance of targets
        #min_score_improv = 0.01/yvar #
        #print(yvar)
        #print(min_score_improv)
        zoom_cond = True
        grid_counter = 0
        # Splitting method
        # Splitting method 
        if ts_split:
            split_method = TSSplit(n_splits=n_splits)
        else:
            split_method = n_splits
        
        # print('GRID COUNTEER ',grid_counter)
        #create hyperparameters grid
        kvalues = np.ceil(np.linspace(kmin, kmax,knum)).astype('int')
        # Remove duplicates
        kvalues = list(set(kvalues))
        # Fit KNN with parameter selection based on tssplit cross validation
        param_grid = {"n_neighbors": kvalues}
        knn = GridSearchCV(KNeighborsRegressor(), 
                          param_grid=param_grid,
                          cv = split_method,
                          refit = True, 
                          return_train_score = True)
        knn.fit(X,Y)
        # print("Time for KRR fitting: %.3f" % (time.time() - stime))
        df = pd.DataFrame(data = knn.cv_results_)
        cv.append(df)
        est = knn.best_estimator_ # the best model and parameters
        best_k = est.get_params()['n_neighbors']
        best_k_index = np.where(kvalues==best_k)[0][0]
        best_std_valid_score = df.loc[knn.best_index_,'std_test_score']
        best_mean_valid_score = df.loc[knn.best_index_,'mean_test_score']
        
        self.estimator=est
        self.cv=cv
        self.fit(idx_start=idx_start,idx_end=idx_end,add2history=add2history)
        
        if get_info:
            return X,Y,cv    
    
    # =========================================================================
    # Utilities
            
    # =========================================================================  
    def set_params(self, parameters):
        """
        Set the parameters of the estimator
        """
        self.estimator.set_params(**parameters)
        return self

    
    def get_params(self):
        """
        Get the parameters of the estimator
        """
        return self.estimator.get_params()


    def update_history(self):
        '''
        Update the history of the model
        '''
        self.history_estimators.append(deepcopy(self.estimator))
        self.history_t.append(deepcopy(self.fitted_indices[-1]))
        self.history_d.append(deepcopy(self.d))
        self.history_cv.append(deepcopy(self.cv))
        return self


    # =========================================================================   
    def _set_params_aux(self, keys, pars):
        """
        Auxiliary for cross validation
        """
        for i in range(len(keys)):
            self.estimator.set_params(**{keys[i]: pars[i]})
        return self
    # =========================================================================

       
    







# ============================================================================= 
# Multi horizon autoregressive models
# ============================================================================= 

class AutoRegMH:
    """
    Dictionary of AutoReg for different time horizons 
    Instances:
        - estimator: learning estimator with methods:
                        - .fit 
                        - .predict 
                        - .set_params
        - d: size of the previous lag to consider in the model
        - ls_H: list-like of time horizons
        - series: series to apply the methods to
        - models: dictionary of AutoReg for different horizon with the
                  horizon as keys.
    
    Methods:
        - fit: used to fit the learning methods using a slice of 
               the timese series
        - predict: predict some indices in the time series        
    """
    
# =============================================================================    
    def __init__(self, 
                 estimator=LinearRegression(), 
                 d=1, 
                 max_H=1,
                 ls_H=[] ,
                 series=None):
        """Initilizer"""
        self.estimator=estimator
        self.d = d
        self.series=series
        
        if ls_H:
            self.ls_H = ls_H
        else:
            self.ls_H = list(range(1,max_H+1))
        
        self.models={}    
        for H in self.ls_H: 
            self.models[H] = deepcopy(AutoReg(H=H,
                                              series=self.series,
                                              d=self.d,
                                              estimator=self.estimator
                                              ))
   
    def fit(self,series=None,
            idx_start=0, 
            idx_end=-1, 
            add2history=False):
        """
        Fit all the autoregressive models for the differents horizons.
        """
        if series is None:
            series = self.series.copy()
        
        for H in self.ls_H:
            self.models[H].fit(series=series,
                                idx_start=idx_start, 
                                idx_end=idx_end, 
                                add2history=add2history)
        return self    
# =============================================================================   







    
    
# =============================================================================
# Functions to handle the data
# =============================================================================


def create_dataset(series,
                   idx_start, idx_end,
                   d=1,H=1,vector_y=False):
    '''
    Create the inputs and outputs using a slice of the time series
    Inputs:
      series= series object containing the time series
      idx_start, idx_end= we only consider the indices between these two
      T= number of lag features considered
      H=horizon to predict in the future
      vector_y= boolean denoting if the y is a vector of the next H values or 
                a scalar  with the H-th future value.
    
    Outputs:
      df= dataframe containing all the lag features and the future values to predict
      X= input data matrix of the lag values
      Y= output matrix with the future values to be predicted.
    '''
    # Cut the series from start to end indices
    series=series[idx_start : idx_end]
    # Datframe with Lag variables
    df=pd.DataFrame()
    for t in np.arange(1,d+1,1)[::-1]:
        df['lag_'+str(t-1)]=series.shift(t)
    # If i want vector valued output with next H values
    if vector_y :    
        for t in np.arange(1,H+1,1):     
            df['Future_'+str(t)]=series.shift(-t+1)
        # Remove nan
        df=df.dropna(how='any')
        df.reset_index(drop=True, inplace=True)
        X=df.values[:,:-H]
        Y=df.values[:,-H:]
    # The output is the H-th future value 
    else: 
        df['Future_'+str(H)]=series.shift(-H+1)
        # Remove nan
        df=df.dropna(how='any')
        df.reset_index(drop=True, inplace=True)
        X=df.values[:,:-1]
        Y=df.values[:,-1]
    return df, X, Y 
     

# =============================================================================
def create_inputs(series,
                  idx_start, 
                  idx_end,
                  d=1):
    '''
    Create the input points using a given slice of the time series
    Inputs:
      series= series object containing the time series
      idx_start, idx_end= we only consider the indices between these two
      T= number of lag features considered
    
    Outputs:
      X= input data matrix of the lag values
    '''
    
    # Cut the series from start to end indices
    series=series[idx_start : idx_end]
    # Datframe with Lag variables
    df=pd.DataFrame()
    for t in np.arange(d)[::-1]:
        df['lag_'+str(t-1)]=series.shift(t)
    # Remove nan
    df=df.dropna(how='any')
    X=df.values
    return  X
# =============================================================================






        
        