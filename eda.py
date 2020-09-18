import os
import numpy as np
import pandas as pd
from conversions import m2nm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import ceil


plt.style.use('ggplot')


def get_touchdown(df):
    """
    Calculates instance of touchdown (landing).
    """
    df = df.copy()
    return df.iloc[df['Altitude'].idxmax():][df['Altitude'] <= 2].index[0]


def distance_from_touchdown(df):
    """
    Calculates the distance until the touchdown instant across the entire
    flight.
    """
    df = df.copy()
    df['Distance'] = (df.Time.diff()*df.AirSpeed.cumsum()*m2nm).fillna(0)
    touchdown_point = df['Distance'].iloc[get_touchdown(df)]
    df['DistanceFromTouchdown'] = df['Distance'] - touchdown_point

    return df


def pre_process_flights(flights_folder):
    """
    Imports and merges flight files inside input folder.
    """
    df_flights = pd.DataFrame()
    for flight_file in os.listdir(flights_folder):
        print('Processing flight: '+flight_file)
        df_flight = pd.read_csv(os.path.join(flights_folder, flight_file))
        df_flight['flight_id'] = flight_file.split('.')[0]
        df_flight = distance_from_touchdown(df_flight)
        print(df_flight.head())
        df_flights = df_flights.append(df_flight, ignore_index=True)
    return df_flights


def resample_flights(df, base_column, interval_min, interval_max,
                     samples_per_flight=None):
    """
    Resample flight data based on given interval for the reference columm
    input.
    Used for calculating the quantiles for each parameter during the flight.
    """
    df = df.copy()

    output = pd.DataFrame()

    avg_samples_per_flight = int(len(df)/len(df['flight_id'].unique()))

    if samples_per_flight is None:
        samples_per_flight = avg_samples_per_flight

    resampled_base_column = np.linspace(interval_min, interval_max,
                                        num=samples_per_flight)
    for flight in df['flight_id'].unique():
        intermediate = df[df['flight_id'] == flight]
        intermediate_resampled = pd.DataFrame()
        intermediate_resampled[base_column] = resampled_base_column
        for column in df.columns.values:
            if column == 'flight_id':
                continue
            intermediate_resampled[column] = np.interp(
                    intermediate_resampled[base_column],
                    intermediate[base_column], intermediate[column])
            intermediate_resampled['flight_id'] = flight
        output = output.append(intermediate_resampled.drop_duplicates(),
                               ignore_index=True)

    return output


def get_quantile_per_sample(df, quantile, base_column, object_column=None):
    """
    Calculates the specified quantile accross the sample.
    """
    df = df.copy()
    output = pd.DataFrame()
    for unique_sample in df[base_column].unique():
        sample_size = len(df[df[base_column] == unique_sample])
        if sample_size < 10:
            continue
        object_df = df[df[base_column] == unique_sample]
        output = output.append(object_df.quantile(quantile))
    return output.sort_values(by=base_column)


def plot_quantiles(lower_quantile, upper_quantile):
    sns.set()
    fig, ax = plt.subplots()
    sns.lineplot(data=lower_quantile, x='DistanceFromTouchdown',
                 y='Altitude', ax=ax, color='black')
    sns.lineplot(data=upper_quantile, x='DistanceFromTouchdown',
                 y='Altitude', ax=ax, color='black')
    ax.fill_between(lower_quantile['DistanceFromTouchdown'],
                    lower_quantile['Altitude'],
                    upper_quantile['Altitude'],
                    color="#BBC0C4")
    plt.show()

    return


def plot_flights_n_boundary(df_flights, x_column, y_column,
                            df_lower_boundary=None, df_upper_boundary=None,
                            xlims=None, ylims=None,
                            title=None, xlabel=None, ylabel=None,
                            lower_boundary_label=None,
                            upper_boundary_label=None,
                            highlight_flight=None,
                            highlight_flights=None,
                            label_highlighted_flights=False,
                            include_legend=True):
    """
    Plots pair of parameters for every flight within the compiled dataframe.
    """
    df_flights = df_flights.copy()
    fig, ax = plt.subplots()
    if xlims is None:
        x_min = round(min(df_flights[x_column]), -1)
        x_min_margin = round(0.1*(round(min(df_flights[x_column]), -1)) -
                             ceil(max(df_flights[x_column])), -2)
        x_max = ceil(max(df_flights[x_column]))
        x_max_margin = round(0.1*(ceil(max(df_flights[x_column])) -
                                  round(min(df_flights[x_column]), -1)), -2)
        xlims = (x_min+x_min_margin, x_max+x_max_margin)
    else:
        xlims = sorted(xlims)
        df_flights = df_flights[
                df_flights[x_column] >= xlims[0]][
                        df_flights[x_column] <= xlims[1]]
    if ylims is None:
        ylims = (
                round(min(df_flights[y_column]), -1),
                ceil(max(df_flights[y_column]))
                 )
    else:
        ylims = sorted(ylims)
        df_flights = df_flights[
                df_flights[y_column] >= ylims[0]][
                        df_flights[y_column] <= ylims[1]
                ]
    if xlabel is None:
        xlabel = x_column
    if ylabel is None:
        ylabel = y_column
    if title is None:
        title = y_column+' vs '+x_column
    if lower_boundary_label is None:
        lower_boundary_label = 'Lower boundary'
    if upper_boundary_label is None:
        upper_boundary_label = 'Upper boundary'

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    array_collection = [
            np.transpose(
                    np.column_stack(
                            df_flights[df_flights['flight_id'] == flight_id]
                            [[x_column, y_column]].to_numpy())
                    ) for flight_id in df_flights['flight_id'].unique()
                    ]
    line_segments = LineCollection(array_collection,
                                   linewidths=(0.5, 1, 1.5, 2),
                                   linestyles='solid',
                                   alpha=0.5,
                                   label='Flights')
    line_segments.set_array(np.arange(len(array_collection)))
    ax.add_collection(line_segments)

    if (df_lower_boundary is not None) & (df_upper_boundary is not None):
        _ = ax.plot(df_lower_boundary[x_column],
                    df_lower_boundary[y_column],
                    color='black',
                    linewidth=2.5,
                    label=lower_boundary_label)
        _ = ax.plot(df_upper_boundary[x_column],
                    df_upper_boundary[y_column],
                    color='black',
                    linewidth=2.5,
                    label=upper_boundary_label)
        ax.fill_between(df_lower_boundary[x_column],
                        df_lower_boundary[y_column],
                        df_upper_boundary[y_column],
                        color="black")

    if highlight_flight is not None:
        _ = ax.plot(df_flights[df_flights['flight_id'] ==
                               highlight_flight][x_column],
                    df_flights[df_flights['flight_id'] ==
                               highlight_flight][y_column],
                    color='red',
                    linewidth=1.25,
                    label=highlight_flight)

    if highlight_flights is not None:
        for flight in highlight_flights:
            _ = ax.plot(df_flights[df_flights['flight_id'] ==
                                   flight][x_column],
                        df_flights[df_flights['flight_id'] ==
                                   flight][y_column],
                        color='red',
                        linewidth=1.25,
                        label=flight if label_highlighted_flights else "_nolegend_")

    if include_legend:
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.show()

    return fig, ax


if __name__ == "__main__":
    process_flights = None
    if process_flights is not None:
        flights_folder = './input_flights'
        df_flights = pre_process_flights(flights_folder)
    else:
        df_flights = pd.read_csv('processed_data/compiled_flights.zip')

    df_flights_resampled = resample_flights(
            df_flights, "DistanceFromTouchdown",
            round(min(df_flights["DistanceFromTouchdown"]), -1), 0,
            samples_per_flight=int(len(df_flights) /
                                   len(df_flights['flight_id'].unique()))
            )
    quantile_10 = get_quantile_per_sample(df_flights_resampled, 0.1,
                                          "DistanceFromTouchdown")
    quantile_90 = get_quantile_per_sample(df_flights_resampled, 0.9,
                                          "DistanceFromTouchdown")
    _ = plot_flights_n_boundary(df_flights,
                                'DistanceFromTouchdown', 'Altitude',
                                df_lower_boundary=quantile_10,
                                df_upper_boundary=quantile_90,
                                xlims=(-2, 1), ylims=(0, 6000),
                                title='Altitude vs Distance from touchdown',
                                xlabel='Distance from touchdown [NM]',
                                ylabel='Altitude [ft]',
                                lower_boundary_label='10th percentile',
                                upper_boundary_label='90th percentile'
                                )
