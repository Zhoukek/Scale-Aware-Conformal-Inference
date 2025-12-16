import os, copy
import numpy as np
import torch
import pandas as pd
import warnings
from .ar import generate_process
from darts.models.forecasting.prophet_model import Prophet
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.theta import Theta
from darts.models.forecasting.transformer_model import TransformerModel
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.dlinear import DLinearModel
from darts.models.forecasting.nlinear import NLinearModel
from darts.models.forecasting.tsmixer_model import TSMixerModel
from darts.models.forecasting.xgboost import XGBModel
from darts.models.forecasting.lgbm import LightGBMModel
from darts import TimeSeries
import pdb
from tqdm import tqdm
import random

def generate_forecasts(
    data,
    model_name,
    savename,
    overwrite,
    log,
    fit_every,
    ahead,
    use_gpu=True,
    *args,
    **kwargs
):
    if not overwrite:
        try:
            saved = np.load(savename)
            forecasts = saved["forecasts"]
            return forecasts
        except:
            pass

    input_chunk_length = kwargs.get('input_chunk_length')
    output_chunk_length = kwargs.get('output_chunk_length')
    
    T = data.shape[0]
    forecasts = np.zeros((T,))
    data2 = copy.deepcopy(data)
    
    if log:
        data2['y'] = np.log(data2['y'])
    
    data2.index = pd.date_range(start=data2.index.min(), periods=len(data2), freq='D')
    y = TimeSeries.from_dataframe(data2['y'].interpolate().reset_index(drop=False).sort_values(by='index'), 
                                 time_col='index', value_cols='y')
    
    print(f"Generating forecasts with {model_name}...")
    model = None
    
    if use_gpu:
        if torch.cuda.is_available():
            gpu_device = "gpu"
            print("GPU available, using CUDA")
        else:
            gpu_device = "cpu"
            print("GPU requested but not available, using CPU")
            use_gpu = False
    else:
        gpu_device = "cpu"
        print("Using CPU")
    
    if model_name == "prophet":
        model = Prophet()
    elif model_name == "ar":
        model = ARIMA(p=3, d=0, q=0)
    elif model_name == "theta":
        if fit_every > 1:
            raise ValueError("Theta does not support fit_every > 1")
        model = Theta()
    elif model_name == "transformer":
        model = TransformerModel(           
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=100
        )
        y = y.astype(np.float32)
        
    elif model_name == "linear_regression":
        model = LinearRegressionModel(lags=input_chunk_length, output_chunk_length=output_chunk_length)
    
    elif model_name == "rnn":
        model = RNNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            pl_trainer_kwargs={"accelerator": gpu_device, "devices": [0],} if use_gpu else {},
        )
        y = y.astype(np.float32)
    
    elif model_name == "tcn":
        model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            pl_trainer_kwargs={"accelerator": gpu_device} if use_gpu else {},
        )
        y = y.astype(np.float32)
    
    elif model_name == "dlinear":
        model = DLinearModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            pl_trainer_kwargs={"accelerator": gpu_device} if use_gpu else {},
        )
        y = y.astype(np.float32)
    
    elif model_name == "nlinear":
        model = NLinearModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            pl_trainer_kwargs={"accelerator": gpu_device} if use_gpu else {},
        )
        y = y.astype(np.float32)
    
    elif model_name == "tsmixer":
        model = TSMixerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            pl_trainer_kwargs={"accelerator": gpu_device} if use_gpu else {},
        )
        y = y.astype(np.float32)
    
    elif model_name == "xgboost":
        model = XGBModel(
            lags=input_chunk_length,
            output_chunk_length=output_chunk_length,
            tree_method='gpu_hist' if use_gpu else 'hist',
            predictor='gpu_predictor' if use_gpu else 'cpu_predictor'
        )
    
    elif model_name == "lightgbm":
        model = LightGBMModel(
            lags=input_chunk_length,
            output_chunk_length=output_chunk_length,
            device='gpu' if use_gpu else 'cpu'
        )
    
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 设置重新训练频率
    if model_name in ["transformer", "rnn", "tcn", "dlinear", "nlinear", "tsmixer"]:
        retrain = max(fit_every * 10, 100)
    elif model_name in ["xgboost", "lightgbm"]:
        retrain = fit_every * 2
    else:
        retrain = fit_every
    
    try:
        model_forecasts = model.historical_forecasts(
            y, 
            forecast_horizon=fit_every, 
            retrain=retrain, 
            verbose=True
        ).values()[:,-1].squeeze()
        
        forecasts[-model_forecasts.shape[0]:] = model_forecasts
        
    except Exception as e:
        print(f"Error during forecasting with {model_name}: {str(e)}")
        if kwargs.get('fallback', 'zeros') == 'naive':
            naive_forecasts = np.roll(y.values().squeeze(), fit_every)
            naive_forecasts[:fit_every] = y.values().squeeze()[:fit_every]
            forecasts[-len(naive_forecasts):] = naive_forecasts
        else:
            forecasts = np.zeros((T,))
    
    if log:
        forecasts = np.exp(forecasts)
    
    print("Finished generating forecasts.")
    
    np.savez(savename, forecasts=forecasts)
    return forecasts