#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tsengfanghsuan
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Read in Data and Basic Cleaning
def merge_vti(df):
    vti_etf = pd.read_csv("VTI.csv")
    vti_etf = vti_etf[['Close', 'Adj Close']]
    vti_etf.columns = ['VTI_Close', 'VTI_AdjClose']
    df = pd.concat([df, vti_etf], axis = 1)
    return df

def rename_cols(df):
    df = df.rename(columns = {"Adj Close": "AdjClose"})
    return df

def clean_adjclose(df):
    df['AdjClose'] = abs(df['AdjClose'])
    return df

def clean_ohlc(sight_df):
    sight_df['CloseAdjClose_Diff_Roll'] = (sight_df['Close'] - sight_df["AdjClose"]).rolling(30).median()
    sight_df['High_temp'] = sight_df[['Open', 'High', 'Low']].max(axis = 1)
    sight_df['Low_temp'] = sight_df[['Open', 'High', 'Low']].min(axis = 1)
    sight_df['High'] = sight_df['High_temp']
    sight_df['Low'] = sight_df['Low_temp']
    sight_df = sight_df.drop(['High_temp', 'Low_temp'], axis = 1)
    
    for i in range(0, len(sight_df)):
        if (sight_df['Close'][i] < sight_df['Low'][i]) | (sight_df['Close'][i] > sight_df['High'][i]):
            sight_df['Close'][i] = sight_df['Open'][i]
            sight_df['AdjClose'][i] = sight_df['Close'][i] - sight_df['CloseAdjClose_Diff_Roll'][i]
    
    sight_df = sight_df.drop('CloseAdjClose_Diff_Roll', axis = 1)
    return sight_df

def get_daily_return(df):
    df['AdjClose_Diff'] = df['AdjClose'].diff(periods = -1)
    df['AdjClose_Return'] = (-(df['AdjClose_Diff']) / df['AdjClose']).shift(1)
    
    df['Signal_Diff'] = df['Signal'].diff(periods = -1)
    df['Signal_Return'] = (-(df['Signal_Diff']) / df['Signal']).shift(1)
    
    df = df.drop(['AdjClose_Diff', 'Signal_Diff'], axis = 1)
    return df

def get_return_mean_std(df, col):
    df[col + "_mean_roll"] = df[col].rolling(window).mean()
    df[col + "_std_roll"] = df[col].rolling(window).std()
    
    df[col + "_mean_roll"] = df[col + "_mean_roll"].fillna(method = "bfill")
    df[col + "_std_roll"] = df[col + "_std_roll"].fillna(method = "bfill")
    return df

def flag_outlier(df, col):
    df['flag_outlier'] = (df[col] < df[col + "_mean_roll"] + 3 * df[col + "_std_roll"]) * (df[col] > df[col + "_mean_roll"] - 3 * df[col + "_std_roll"])
    df['flag_outlier2'] = df['flag_outlier'].shift(-1)
    df[col + '_flag_outlier'] = np.where((df['flag_outlier'] == False) & (df['flag_outlier2'] == False), False, True)
    df = df.drop(["flag_outlier", "flag_outlier2"], axis = 1)
    return df

def adjust_adjclose_outlier(sight_df):
    sight_df = get_return_mean_std(sight_df, col = "AdjClose_Return")
    sight_df = flag_outlier(sight_df, col = "AdjClose_Return")
    sight_df['CloseAdjClose_Diff_Roll'] = (sight_df['Close'] - sight_df["AdjClose"]).rolling(30).median()
    sight_df['AdjClose'] = np.where(sight_df['AdjClose_Return_flag_outlier'] == True, sight_df['AdjClose'], sight_df['Close'] - sight_df['CloseAdjClose_Diff_Roll'])
    sight_df = sight_df.drop(['AdjClose_Return_mean_roll', 'AdjClose_Return_std_roll', 'AdjClose_Return_flag_outlier', 'CloseAdjClose_Diff_Roll'], axis = 1)
    return sight_df

def remove_signal_outliers(df):
    df = get_return_mean_std(df, col = 'Signal_Return')
    df = flag_outlier(df, col = "Signal_Return")
    
    df = df.drop(df[(df['Signal_Return_flag_outlier'] == False)].index)
    df = df.drop(['Signal_Return_mean_roll', 'Signal_Return_std_roll', 'Signal_Return_flag_outlier', ], axis = 1)
    return df

def read_and_clean():
    sight_df = pd.read_excel('Sample Dataset.xlsx')
    sight_df = rename_cols(sight_df)
    sight_df = clean_adjclose(sight_df)
    sight_df = clean_ohlc(sight_df)
    return sight_df

def get_return_and_remove_outliers(sight_df):
    sight_df = get_daily_return(sight_df)
    sight_df = adjust_adjclose_outlier(sight_df)
    sight_df = remove_signal_outliers(sight_df)
    sight_df = get_daily_return(sight_df)
    
    outlier_flag = True
    while outlier_flag == True:
        n1 = len(sight_df)
        sight_df = remove_signal_outliers(sight_df)
        sight_df = adjust_adjclose_outlier(sight_df)
        n2 = len(sight_df)
        if n1 == n2:
            outlier_flag = False

    return sight_df


# 2. Statictis to check signal
def get_rolling_corr(df, col1, col2):
    df[col1+"_"+col2+"_rolling_corr"] = df[col1].rolling(window).corr(df[col2])
    return df

def get_direct_correct(df, col1, col2):
    df[col1 + "_Dir_Correct"] = df[col1] * df[col2] > 0
    df[col1 + "_Dir_Correct"] = df[col1 + "_Dir_Correct"].astype(int)
    df[col1 + "_Dir_Correct_Roll"] = df[col1 + "_Dir_Correct"].rolling(window).sum() / window
    return df

def run_corr_dircorrect(sight_df):
    sight_df = get_rolling_corr(sight_df, col1 = 'Signal', col2 = 'AdjClose')
    sight_df = get_rolling_corr(sight_df, col1 = 'Signal_Return', col2 = 'AdjClose_Return')
    
    sight_df = get_direct_correct(sight_df, col1 = 'Signal_Return', col2 = 'AdjClose_Return')
    return sight_df

def get_outsample_r2(sight_df, col1, col2):
    out_r2 = [None] * 252 * 2
    for i in range(window, len(sight_df) - window):
        out_r2_score = r2_score(sight_df[col1][i:i+window], sight_df[col2][i:i+window])
        out_r2.append(out_r2_score)
    sight_df[col2 + "_out_r2"] = out_r2
    return sight_df

# 3. Models to Predict Return

# Build Rolling Regression Model
def build_return_model(sight_df):
    sight_df['const'] = 1
    model = RollingOLS(endog = sight_df['AdjClose_Return'].values, exog = sight_df[['const', 'Signal_Return']], window = 252)
    reg = model.fit()
    
    model_coefs = reg.params
    model_coefs.columns = ['intercept_return', 'coef_signal_return']
    sight_df = pd.concat([sight_df, model_coefs], axis = 1)
    
    sight_df['ols_intercept_return'] = sight_df['intercept_return'].shift(1)
    sight_df['ols_coef_return'] = sight_df['coef_signal_return'].shift(1)
    sight_df['ols_pred_return'] = sight_df['ols_intercept_return'] + sight_df['ols_coef_return'] * sight_df['Signal_Return']
    sight_df = sight_df.drop(['const'], axis = 1)
    return sight_df

def build_price_model(sight_df):
    sight_df['const'] = 1
    model = RollingOLS(endog = sight_df['AdjClose'].values, exog = sight_df[['const', 'Signal']], window = 252)
    reg = model.fit()
    
    model_coefs = pd.concat([reg.params, reg.rsquared], axis = 1)
    model_coefs.columns = ['ols_intercept_price', 'ols_coef_signal', 'ols_insample_r2']
    sight_df = pd.concat([sight_df, model_coefs], axis = 1)
    
    sight_df['ols_intercept_price'] = sight_df['ols_intercept_price'].shift(1)
    sight_df['ols_coef_signal'] = sight_df['ols_coef_signal'].shift(1)
    sight_df['ols_pred_price'] = sight_df['ols_intercept_price'] + sight_df['ols_coef_signal'] * sight_df['Signal']
    sight_df['ols_pred_price_return'] = (sight_df['ols_pred_price'] - sight_df['AdjClose'].shift(1)) / sight_df['AdjClose'].shift(1)
    sight_df = sight_df.drop(['const'], axis = 1)
    return sight_df

# Build Rolling Ridge Model
def build_ridge_price_model(sight_df):
    model_intercept_result = [None] * 252
    model_coef_result = [None] * 252
    model_r2_result = [None] * 252
    
    for i in range(252, len(sight_df)):
        print(i)
        ridge_model = Ridge()
        alpha_values = np.linspace(50, 55, 5)
        grid = GridSearchCV(estimator = ridge_model, param_grid = dict(alpha = alpha_values))
        grid.fit(sight_df['Signal'][i-252:i].values.reshape(-1, 1), sight_df['AdjClose'][i-252:i])
        model_intercept_result.append(grid.best_estimator_.intercept_)
        model_coef_result.append(grid.best_estimator_.coef_[0])
        model_r2_result.append(r2_score(sight_df['AdjClose'][i-252:i], grid.predict(sight_df['Signal'][i-252:i].values.reshape(-1, 1))))
        
    sight_df['ridge_intercept_price'] = model_intercept_result
    sight_df['ridge_coef_price'] = model_coef_result
    sight_df['ridge_insample_r2'] = model_r2_result
    
    sight_df['ridge_pred_price'] = sight_df['ridge_intercept_price'] + sight_df['Signal'] * sight_df['ridge_coef_price']
    sight_df['ridge_pred_price_return'] = (sight_df['ridge_pred_price'] - sight_df['AdjClose'].shift(1)) / sight_df['AdjClose'].shift(1)
    return sight_df

# Build Models
def build_all_models(sight_df):
    sight_df = build_return_model(sight_df)
    sight_df = build_price_model(sight_df)
    sight_df = build_ridge_price_model(sight_df)
    
    return sight_df

# Rolling Correlation
def get_model_metrics(sight_df):
    sight_df = get_rolling_corr(sight_df, col1 = 'ols_pred_return', col2 = 'AdjClose_Return')
    
    sight_df = get_rolling_corr(sight_df, col1 = 'ols_pred_price', col2 = 'AdjClose')
    sight_df = get_rolling_corr(sight_df, col1 = 'ols_pred_price_return', col2 = 'AdjClose_Return')
    
    sight_df = get_rolling_corr(sight_df, col1 = 'ridge_pred_price', col2 = 'AdjClose')
    sight_df = get_rolling_corr(sight_df, col1 = 'ridge_pred_price_return', col2 = 'AdjClose_Return')
    
    # Direction Correctness
    sight_df = get_direct_correct(sight_df, col1 = 'ols_pred_price_return', col2 = 'AdjClose_Return')
    sight_df = get_direct_correct(sight_df, col1 = 'ridge_pred_price_return', col2 = 'AdjClose_Return')
    
    # R2 Score
    sight_df = get_outsample_r2(sight_df, col1 = 'AdjClose', col2 = 'ols_pred_price')
    sight_df = get_outsample_r2(sight_df, col1 = 'AdjClose', col2 = 'ridge_pred_price')
    return sight_df


# 4. Investment Strategy Back-Test
def build_investment_portfolio(sight_df):
    investment_df = sight_df[['Date', 'Signal', 'AdjClose', 'AdjClose_Return', 'ols_pred_price_return']][253:1027]
    investment_df['model_return'] = investment_df['ols_pred_price_return'] > 0
    investment_df['model_return'] = np.where(investment_df['model_return'] == True, 1, -1)
    investment_df['model_return'] = investment_df['model_return'] * investment_df['AdjClose_Return']
    
    investment_df['base_portfolio'] = (investment_df["AdjClose_Return"] + 1).cumprod()
    investment_df['model_portfolio'] = (investment_df['model_return'] + 1).cumprod()
    
    investment_df['rolling_return_model_portfolio'] = investment_df['model_return'].rolling(window).apply(lambda x: (x + 1).prod() - 1)
    investment_df['rolling_return_base_portfolio'] = investment_df['AdjClose_Return'].rolling(window).apply(lambda x: (x + 1).prod() - 1)
    investment_df['rolling_std_model_portfolio'] = investment_df['rolling_return_model_portfolio'].rolling(window).std()
    investment_df['rolling_std_base_portfolio'] = investment_df['rolling_return_base_portfolio'].rolling(window).std()
    
    return investment_df


def run_all():
    sight_df = read_and_clean()
    sight_df = get_return_and_remove_outliers(sight_df)
    sight_df = run_corr_dircorrect(sight_df)
    sight_df = build_all_models(sight_df)
    sight_df = get_model_metrics(sight_df)
    
    investment_df = build_investment_portfolio(sight_df)
    return sight_df, investment_df

window = 252
sight_df, investment_df = run_all()







