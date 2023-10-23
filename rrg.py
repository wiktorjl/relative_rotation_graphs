import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors

STOCKS = ['XLE', 'XLK', 'XLU', 'XLV', 'XLF', "XLC"]

BENCHMARK = 'SPY'

BASIC_COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "cyan", "magenta", "lime", "teal", "indigo", "maroon", "navy", "olive", "silver",
    "aqua", "fuchsia", "lightblue", "darkgreen", "coral", "gold", "ivory", "khaki", "orchid", "plum", "salmon"
]

COLORS = {stock: BASIC_COLORS[index % len(BASIC_COLORS)] for index, stock in enumerate(STOCKS)}

TAIL_LENGTH = 10


def set_background_colors(ax):
    lightred = mcolors.to_rgba('red', alpha=0.2)
    ax.add_patch(patches.Rectangle((90, 100), 10, 10, facecolor='lightblue'))
    ax.add_patch(patches.Rectangle((100, 100), 10, 10, facecolor='lightgreen'))
    ax.add_patch(patches.Rectangle((90, 90), 10, 10, facecolor=lightred))
    ax.add_patch(patches.Rectangle((100, 90), 10, 10, facecolor='lightyellow'))
    ax.text(90.5, 109.2, 'Improving', fontsize=12, color='black')
    ax.text(100.6, 109.2, 'Leading', fontsize=12, color='black')
    ax.text(90.5, 90.5, 'Lagging', fontsize=12, color='black')
    ax.text(100.6, 90.5, 'Weakening', fontsize=12, color='black')

def draw_rrg_paths(ax, data_points):
    for label, coords_list in data_points.items():
        if len(coords_list) > 1:
            x_values, y_values = zip(*coords_list)
            t = np.linspace(0, 1, len(x_values))
            t_smooth = np.linspace(0, 1, 300)

            window_length = min(2 * (len(x_values) // 2) + 1, len(x_values))
            x_smooth_values = savgol_filter(x_values, window_length, 3)
            y_smooth_values = savgol_filter(y_values, window_length, 3)

            spline_x = make_interp_spline(t, x_smooth_values, k=1)
            spline_y = make_interp_spline(t, y_smooth_values, k=1)
            x_smooth = spline_x(t_smooth)
            y_smooth = spline_y(t_smooth)

            ax.plot(x_smooth, y_smooth, zorder=1) 

            for i, coords in enumerate(zip(x_smooth_values, y_smooth_values)):
                if i == len(coords_list) - 1:
                    color = 'red'
                    ax.annotate(label, (coords[0]+0.1, coords[1]+0.1), fontsize=10)
                else:
                    color = COLORS[label]

                ax.scatter(*coords, c=color, zorder=2)

def read_and_preprocess_data(filename):
    prices = pd.read_csv(filename, index_col=['date', 'code'])
    prices = prices.drop(["exchange_short_name", "open", "low", "high", "volume", "EOD_Sector", "EOD_Industry"], axis=1)
    prices.sort_values(by='date', inplace=True)
    prices = prices.unstack(level=-1)
    prices.columns = prices.columns.droplevel()
    prices.index = pd.to_datetime(prices.index, format='%Y-%m-%d')
    prices = prices.resample('W').last()

    return prices


def create_rrg_graph(data_points, start_date, end_date, filename='rrg.png'):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(90, 110)
    ax.set_ylim(90, 110)
    ax.set_xticks(np.arange(90, 110, 1))
    ax.set_yticks(np.arange(90, 110, 1))
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Momentum')
    ax.set_title('RRG Graph', fontsize=14, loc='left')
    ax.set_title(f'{start_date} - {end_date}', fontsize=12, loc='right')

    set_background_colors(ax)
    draw_rrg_paths(ax, data_points)

    plt.savefig(filename, dpi=300)
    plt.close()


def calculate_rss_and_momentum(stock_data, stocks):
    rsss = {}
    moms = {}
    momentum_period = 5
    
    for column in stocks:
        if column != 'date':

            # This is just relative strength: price of stock / price of benchmark
            stock_data[f"RS_{column}"] = stock_data[column] / stock_data[BENCHMARK]

            # We have a fast and a slow EMA for smoothing the RS
            stock_data[f"EMA10_RS_{column}"] = stock_data[f"RS_{column}"].ewm(span=10, adjust=False).mean()
            stock_data[f"EMA30_RS_{column}"] = stock_data[f"RS_{column}"].ewm(span=30, adjust=False).mean()
            
            # This is the smoothing step. Multiplied by 100 so it displays nicely on the graph
            stock_data[f"RS_Ratio_{column}"] = (stock_data[f"EMA10_RS_{column}"] / stock_data[f"EMA30_RS_{column}"]) * 100
            rsss[f"{column}"] = stock_data[f"RS_Ratio_{column}"]


            # Momentum is simply price now / price some periods ago.
            mom = rsss[f"{column}"].diff(periods=momentum_period) * 100
            min_val = mom.min()
            max_val = mom.max()

            # Normalize the momentum values to 90-110 range
            mom = 20 * (mom - min_val) / (max_val - min_val) + 90
            moms[f"{column}"] = mom

    return rsss, moms


if __name__ == "__main__":

    # Read the raw input file...
    raw_data = read_and_preprocess_data('/data/output/adjusted_combined.csv')
    # ... and calculate relative strength (RS) and momentum values
    rss, moms = calculate_rss_and_momentum(raw_data, STOCKS)

    # Merge the data for display
    data = {}
    for stock in STOCKS:
        data[stock] = list(zip(rss[stock], moms[stock]))[-TAIL_LENGTH:]

    # Since we want to show the date range of the graph, extract min and max date from the first element
    min_dt = rss[list(rss.keys())[0]].index[-TAIL_LENGTH]
    max_dt = rss[list(rss.keys())[0]].index[-1]
    create_rrg_graph(data, min_dt, max_dt, 'rrg.png')
