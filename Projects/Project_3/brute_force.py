""" This script uses a brute force method to find the best combination of 
columns to use in a linear regression model. """

import pathlib
import statsmodels.api as sm
import pandas as pd
from tqdm import tqdm

def make_dummies(df: pd.DataFrame, target:str = 'y') -> pd.DataFrame:
    """
    Creates dummy variables for categorical variables in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target (str): The target variable column name. Default is 'y'.

    Returns:
    - df (DataFrame): The modified DataFrame with dummy variables.

    Example:
    >>> df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green'], 'Size': ['Small', 'Medium', 'Large']})
    >>> df = make_dummies(df)
    >>> print(df)
       Color_Blue  Color_Green  Color_Red  Size_Large  Size_Medium  Size_Small
    0           0            0          1           0            0           1
    1           1            0          0           0            1           0
    2           0            1          0           1            0           0
    """

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object' and col != target:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=int)], axis=1)
            df.drop(col, axis=1, inplace=True)

    return df

def find_best(df:pd.DataFrame, start: str, num_cols: int = 5, target: str = 'y') -> tuple:
    """ Given a DataFrame, a starting column, and a number of columns to test, returns the 
    best columns to use in a linear regression model."""

    best_cols = [start]

    remaining_cols = df.drop([target, start], axis=1).columns.to_list()

    while len(best_cols) < num_cols:
        testing_cols = pd.DataFrame(columns=['rsquared'])

        for col in remaining_cols:
            cols = best_cols + [col]
            rsquared = sm.OLS(df[target], sm.add_constant(df[cols])).fit().rsquared
            testing_cols.loc[col] = [rsquared]


        best = testing_cols.idxmax()[0]
        best_cols.append(best)
        remaining_cols.remove(best)

    return (best_cols, sm.OLS(df[target], sm.add_constant(df[best_cols])).fit().rsquared)

def run_test(path, target: str = "y", n_cols: int = 5):
    """ Runs a test on the given file, target variable, and number of columns. """

    odf = pd.read_csv(path).drop(['duration'], axis=1)
    odf['y'] = odf['y'].map({'yes': 1, 'no': 0})

    odf_dummies = make_dummies(odf)

    cur_max, best_comb, best_start = 0, None, None
    test_cols = odf_dummies.drop('y', axis=1).columns.to_list()

    for _, col_name in tqdm(enumerate(test_cols), total=len(test_cols), desc='Testing columns', unit='col', leave=False):
        comb, rsq = find_best(odf_dummies, col_name, n_cols, target)
        if rsq > cur_max:
            cur_max = rsq
            best_start = col_name
            best_comb = comb

    return (best_comb, best_start, cur_max)

if __name__ == '__main__':
    csv_path = pathlib.Path.cwd() / 'Projects' / 'Project_3' / 'bankdata_training.csv'

    bcomb, bstart, bmax = run_test(csv_path, 'y', 5)

    print(f'Best combination: {bcomb}')
    print(f'Best starting column: {bstart}')
    print(f'Best R-squared: {bmax}')
