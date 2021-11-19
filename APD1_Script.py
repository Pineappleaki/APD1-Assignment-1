"""
Data obtained form: https://www.kaggle.com/amritpal333/crypto-mining-data
Analyse the transactions comprised on the blockchain and compare ethereum to
bitcoin
"""
from datetime import date
from colorama import Fore, Style
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
FUNCTIONS
"""


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
            result[feature_name] = ((df[feature_name] - min_value) /
                                    (max_value - min_value))
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
    print(f'\n*** {Fore.YELLOW + Style.BRIGHT}STARTING:{Fore.RESET}',
          'plotTimeSeries ***')
    # Unpacking lists
    try:
        gen_title, x_label, y1_label, y2_label = labels
    except:
        print(f"{Fore.RED + Style.BRIGHT}ERROR:{Fore.RESET} Labels is missing",
              f"{(4 - len(labels))} parameters")
        print(f'*** {Fore.YELLOW + Style.BRIGHT}ENDING:{Fore.RESET}',
              'plotTimeSeries ***\n')
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

        _plt = ax1.plot(df.index, df[y1])

        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_label)

        _plt2 = ax2.plot(df.index, df[y2], color=color_set[1])

        plt.legend([_plt[0], _plt2[0]], [y1_label, y2_label])
        plt.show()

        if save_file is True:
            file_name = (f'./plots/{title}_{current_date}.png')
            print(f'{Fore.MAGENTA + Style.BRIGHT}SAVING:{Fore.RESET} ',
                  f'{file_name}')
            fig.savefig((file_name),
                        format='png',
                        dpi=120,
                        bbox_inches='tight')

        if produce_summary is True:
            print(f'\nProducing Summary for {Fore.YELLOW + Style.BRIGHT}',
                  f'{asset_set[itn-1]}:{Fore.RESET}\n')
            all_features = [y1, y2]
            total_summary = []
            # Basic summary
            for feature in all_features:
                summary = []
                summary.append(np.max(df[feature]))
                summary.append(np.min(df[feature]))
                summary.append(np.mean(df[feature]))

                total_summary.append(summary)
            feature_df = pd.DataFrame(total_summary,
                                      columns=(['Maximum', 'Minimum', 'Mean']),
                                      index=(all_features))
            feature_df = feature_df.transpose()

            spearman = stats.spearmanr(df[y1], df[y2])
            print(feature_df)
            print('\nSpearman R Coefficient:\n',
                  f'{Fore.BLUE + Style.BRIGHT}Correlation:{Fore.RESET} ',
                  f'{spearman[0]:.4f}\n',
                  f'{Fore.BLUE + Style.BRIGHT}P-Value:{Fore.RESET} ',
                  f'{spearman[1]}\n')

    print(f'*** {Fore.YELLOW + Style.BRIGHT}ENDING:{Fore.RESET} ',
          'plotTimeSeries ***\n')
    return


def plotScatter(data, y1, y2='',
                assets='',
                color_set=['#2077b4', '#f38043'],
                save_file=False,
                stats_test=True):
    print(f'\n*** {Fore.YELLOW + Style.BRIGHT}STARTING:{Fore.RESET} ',
          'plotScatter ***\n')

    current_date = date.today().strftime('%d-%m-%Y')
    # Making inputs usable depnding on their value
    if (y2 == '') is True:
        y2 = y1

    if ((isinstance(data, list)) is True):
        df_1, df_2 = data[0], data[1]
    else:
        df_1, df_2 = data, data
        if y1 == y2:
            print(f'{Fore.MAGENTA + Style.BRIGHT}WARNING:{Fore.RESET}',
                  f"You are plotting '{y1}' against it's self.")

    if assets != '':
        plot_labels = True
        if ((isinstance(assets, list)) is True):
            a1, a2 = assets[0], assets[1]
            pass
        else:
            a1 = assets
            a2 = a1
            pass
    else:
        plot_labels = False

    color_a, color_b = color_set
    title = f'{a1}: {y1} vs {a2}: {y2}'

    # Plotting the data:
    plt.figure()
    scatter = sns.regplot(x=df_1[y1], y=df_2[y2],
                          scatter_kws={"color": color_a},
                          line_kws={"color": color_b})
    if plot_labels is True:
        plt.title(title)
        plt.xlabel(f'{a1} {y1}')
        plt.ylabel(f'{a2} {y2}')
    plt.show()

    if save_file is True:
        file_name = (f'./plots/{title}_{current_date}.png')
        print(f'{Fore.MAGENTA + Style.BRIGHT}SAVING:{Fore.RESET} {file_name}')
        fig = scatter.get_figure()
        fig.savefig((file_name),
                    format='png',
                    dpi=120,
                    bbox_inches='tight')

    if stats_test is True:
        spearman = stats.spearmanr(df_1[y1], df_2[y2])
        print(title)
        print('Spearman R Coefficient:\n',
              f'{Fore.BLUE + Style.BRIGHT}Correlation:{Fore.RESET} ',
              f'{spearman[0]:.4f}\n',
              f'{Fore.BLUE + Style.BRIGHT}P-Value:{Fore.RESET} {spearman[1]}')
        ks2 = stats.ks_2samp(df_1[y1], df_2[y2])
        print('KS2 Test:\n',
              f'{Fore.BLUE + Style.BRIGHT}Statistic:{Fore.RESET} ',
              f'{ks2[0]:.4f}\n',
              f'{Fore.BLUE + Style.BRIGHT}P-Value:{Fore.RESET} {ks2[1]}\n')

    print(f'*** {Fore.YELLOW + Style.BRIGHT}ENDING:{Fore.RESET} ',
          'plotScatter ***\n')
    return

# First we need to retrieve the files we will be working with
btc_file = './data/bitcoin.csv'
eth_file = './data/ethereum.csv'

# Load csv into pandas df
btc = pd.read_csv(btc_file)
eth = pd.read_csv(eth_file)

dates = ['2016-08-07', '2018-08-07']
data = matchDates([btc, eth], dates)
btc, eth = data[0], data[1]

# Obtaining weekly average.
btc_weekly = btc.resample('W').mean()
eth_weekly = eth.resample('W').mean()

data1 = btc_weekly
data2 = eth_weekly

data = [data1, data2]
plot_labels = ['Weekly Price & TX Volume', 'Date', 'Price (USD)',
          'Active Addresses']
asset = ['BTC', 'ETH']

plotTimeSeries(data, 'price(USD)', 'adjustedTxVolume(USD)', plot_labels,
               save_file=(False), produce_summary=(True))

plotTimeSeries(data, 'price(USD)', 'activeAddresses', plot_labels,
               save_file=(False), produce_summary=(True))

plotScatter(data2, 'price(USD)', 'activeAddresses', assets = 'ETH',
            save_file=(False))

plotScatter(data, 'price(USD)', 'price(USD)', assets = ['BTC', 'ETH'],
            save_file=(False))

