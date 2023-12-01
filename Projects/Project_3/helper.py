""" Helper functions for Project 3"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import numpy.typing as npt
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleans the dataframe by dropping values in object 
    columns that are `unknown`"""

    cdf = df.copy()
    for col in cdf.columns:
        if cdf[col].dtype == 'object':
            cdf = cdf[cdf[col] != 'unknown']

    return cdf

def le_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """ Creates `sklearn.preprocessing.LabelEncoder` dummy 
    variables for all categorical features in the dataframe"""

    target = 'y'
    categorical_features = []
    numeric_features = []

    # Get all column names
    features = df.columns.to_numpy().tolist()

    # Separate categorical and numeric features
    for col in features:
        if df[col].dtype != 'object':
            if col != target:
                numeric_features.append(col)
        else:
            categorical_features.append(col)

    for col in categorical_features:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).values))
        df[col] = le.transform(list(df[col].astype(str).values))

    return df

def rsquared_abs_corr(df: pd.DataFrame, abs_val: bool = True) -> pd.DataFrame:
    """ Returns a dataframe with the rsquared and absolute 
    correlation of each feature using `pandas.get_dummies` 
    for categorical features"""

    rsquared = pd.DataFrame(columns=['rsquared'])
    target = 'y'

    for col in df.drop('y', axis=1).columns:
        x_values = df[col]

        if df[col].dtype == 'object':
            x_values = pd.get_dummies(x_values, dtype=int)
            x_values = x_values.drop(x_values.columns[0], axis=1)

        y = df[target]
        x_values = sm.add_constant(x_values)
        model = sm.OLS(y, x_values).fit()
        rsquared.loc[col] = model.rsquared

    if abs_val:
        corr = pd.DataFrame(abs(df.corr()['y'])).drop('y')

        corr.rename(columns={'y': 'abs_corr'}, inplace=True)
    
    else:
        corr = pd.DataFrame(df.corr()['y']).drop('y')

        corr.rename(columns={'y': 'corr'}, inplace=True)


    full = rsquared.join(corr, how='inner')
    
    full.sort_values(by=['abs_corr'], ascending=False, inplace=True)

    return full

def corr_iterator(df: pd.DataFrame, target='y', force_clean=False) -> pd.DataFrame:
    """ Adds features to the model one at a time from greater to least 
    correlation and returns the rsquared for each iteration"""

    results = pd.DataFrame(columns=['corr_req', 'rsquared'])

    corr = rsquared_abs_corr(df.copy())

    for req in corr['abs_corr']:
        cols = corr[corr['abs_corr'] >= req].index.to_numpy()

        # need to create a temporary dataframe to avoid length mismatch if force_clean is True
        cur_df = df[[*cols, target]]

        if force_clean:
            cur_df = clean_dataframe(cur_df)

        y = cur_df[target]
        x = le_dummies(df[cols])

        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()

        results.loc[req] = [req, model.rsquared]

    return results
