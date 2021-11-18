"""
Data obtained form: https://www.kaggle.com/amritpal333/crypto-mining-data
Analyse the transactions comprised on the blockchain and compare ethereum to bitcoin
"""
import pandas as pd
import numpy as np

def matchDates(data, date):
    """
    Matches a set of data to the same dates using boolean indexing

    Parameters
    ----------
    data : List
        An array of all the data needed.
    date : List
        Takes two values in standard Y%-M%-D% format as strings.

    Returns
    -------
    return_array : List
        Data array with corrected dates and date in index.

    """
    return_array = []

    for df in data:
        df['date'] = pd.to_datetime(df['date'])
        date1 = [(df[~(df['date'] < date[0])])]
        for df_t in date1:
            date2 = (df_t[~(df_t['date'] > date[1])])
            finalised_df = pd.DataFrame(date2)
            finalised_df = finalised_df.set_index('date')

            return_array.append(finalised_df)

    return return_array


def normalise(df):
    """
    Normalise a given data frame

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe to be normalised.

    Returns
    -------
    result : Pandas Dataframe
        Returns normalised pandas dataframe.

    """
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name == 'date'):
            pass
        else:
            max_value = np.max(df[feature_name])
            min_value = np.max(df[feature_name])
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# First we need to retrieve the files we will be working with
btc_file = './data/bitcoin.csv'
eth_file = './data/ethereum.csv'

# Load csv into pandas df
btc = pd.read_csv(btc_file)
eth = pd.read_csv(eth_file)

dates = ['2016-08-07', '2018-08-07']
data = matchDates([btc, eth],dates)
btc, eth = data[0], data[1]

# Obtaining weekly average.
btc_weekly = btc.resample('W').mean()
eth_weekly = eth.resample('W').mean()

# Normalising data
btc_norm = normalise(btc_weekly)
eth_norm = normalise(eth_weekly)
