"""
Data obtained form: https://www.kaggle.com/amritpal333/crypto-mining-data
Analyse the transactions comprised on the blockchain and compare ethereum to bitcoin
"""
from datetime import date
from colorama import Fore, Style
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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


def plotTimeSeries(data, y1, y2,
                   labels=['Title', 'X-axis', 'Y-axis(left)', 'Y-axis(right)'],
                   asset_set=['BTC', 'ETH'],
                   color_set=['#2077b4', '#f38043'],
                   save_file=False,
                   produce_summary=False):
    """
    Parameters
    ----------
    data : List
        An array of the desired datasets to plot.
    y1 : String
        Left Y axis value.
    y2 : String
        Right Y axis value.
    labels : List
        list of labels: Title;X;Y1;Y2, enter '' to skip a given element.
    asset_set : List
        set of each currency being plotted.
    color_set : List
        set of two colours.
    save_file : Bool
        Save file as a .png, default = False.
    produce_summary : Bool
        Produces summary stats (Min, Max, Mean) and the spearman between
        y1 & y2, default = True.

    Returns
    -------
    Shows a plot for each respective element in 'data', with option to save
    graphics and produce a summary
    """
    print(f'\n*** {Fore.YELLOW + Style.BRIGHT}STARTING:{Fore.RESET} plotTimeSeries ***')
    # Unpacking lists
    try:
        gen_title, x_label, y1_label, y2_label = labels
    except:
        print(f"{Fore.RED + Style.BRIGHT}ERROR:{Fore.RESET} Labels is missing {(4 - len(labels))} parameters")
        print(f'*** {Fore.YELLOW + Style.BRIGHT}ENDING:{Fore.RESET} plotTimeSeries ***\n')
        return
    color_a, color_b = color_set

    current_date = date.today().strftime('%d-%m-%Y')

    # Establishing variables
    itn = 0

    for df in data:

        title = f'{asset_set[itn]}: {gen_title}'
        itn += 1

        fig, ax1 = plt.subplots()

        ax1.tick_params(axis='x', rotation=45)
        ax1.title.set_text(title)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y1_label)
        ax1.grid(True)
        ax1.fill_between(df.index, df[y1], alpha=0.25)

        ax1.plot(df.index, df[y1], label=y1_label)

        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_label)

        ax2.plot(df.index, df[y2], color=color_set[1])

        plt.show()

        if save_file == True:
            file_name = (f'./plots/{title}_{current_date}.png')
            print(f'{Fore.MAGENTA + Style.BRIGHT}SAVING:{Fore.RESET} {file_name}')
            fig.savefig((file_name),
                        format='png',
                        dpi=120,
                        bbox_inches='tight')

        if produce_summary == True:
            print(f'\nProducing Summary for {Fore.YELLOW + Style.BRIGHT}{asset_set[itn-1]}:{Fore.RESET}\n')
            all_features = [y1, y2]
            total_summary = []
            # Basic summary
            for feature in all_features:
                summary = []
                summary.append(np.max(df[feature]))
                summary.append(np.min(df[feature]))
                summary.append(np.mean(df[feature]))

                total_summary.append(summary)
            feature_df = pd.DataFrame(total_summary, columns=(['Maximum', 'Minimum', 'Mean']),
                                      index=(all_features))
            feature_df = feature_df.transpose()

            spearman = stats.spearmanr(df[y1], df[y2])
            print(feature_df)
            print('\nSpearman R Coefficient:\n',
                  f'{Fore.BLUE + Style.BRIGHT}Correlation:{Fore.RESET} {spearman[0]}\n',
                  f'{Fore.BLUE + Style.BRIGHT}P-Value:{Fore.RESET} {spearman[1]}\n')

    print(f'*** {Fore.YELLOW + Style.BRIGHT}ENDING:{Fore.RESET} plotTimeSeries ***\n')
    return


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

# Normalising data - No longer needed.
#btc_norm = normalise(btc_weekly)
#eth_norm = normalise(eth_weekly)

data1 = btc_weekly
data2 = eth_weekly

data = [data1, data2]
plot_labels = ['Weekly Price & TX Count', 'Date', 'Price (USD)',
          'TX Count']
asset = ['BTC', 'ETH']

plotTimeSeries(data, 'price(USD)', 'txCount', plot_labels)