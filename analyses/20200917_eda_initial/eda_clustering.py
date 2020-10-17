import os, inspect
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from eda import get_touchdown, plot_flights_n_boundary, get_quantile_per_sample


def prepare_data(flights_folder):
    """
    Reads flight data and returns a dataframe containing flight id and the
    flight vector x.

    x = [x_1_t1,x_1_t2,...,x_1_tn,...,x_i_tj,...,x_m_t1,x_m_t2,...,x_m_tn]

    where:
        x_i_tj  <- value of the i-th flight parameter at time tj.
        m       <- total number of parameters
        n       <- number of samples for every parameter

    Similar to Cluster AD Flight.
    """
    flights_list = os.listdir(flights_folder)

    flight_vector = []
    for flight in flights_list:
        df_flight = pd.read_csv(os.path.join(flights_folder, flight))
        touchdown_index = get_touchdown(df_flight)
        df_analysis = df_flight[touchdown_index-600:touchdown_index]
        column_filter = df_flight.columns.values[df_flight.columns != 'Time']
        flight_vector.append(
                pd.melt(df_analysis.loc[:, column_filter])['value'].to_numpy()
                )

    return pd.DataFrame({'flight': flights_list, 'vector': flight_vector})


#############################
# 1 - Inputs
#############################
currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe()
        )
    )
)

flights_folder = os.path.join(os.path.dirname(os.path.dirname(currentdir)), 'input_flights')
prepared_data = prepare_data(flights_folder)

#############################
# Preparing data for clustering
#############################
"""
Vale a pena utilizar algum normalizador como StandardScaler().fit_transform()?
Talvez ajude a definir um valor apenas de eps (ver 2)
"""
# Formatting data as numpy array: one flight per row
X = np.empty([len(prepared_data.vector.to_numpy()), 9000])
i = 0
for nparray in prepared_data.vector.to_numpy():
    X[i, :] = nparray
    i += 1

#############################
# 2 - Clustering per se
#############################
"""
# Como calibrar o valor de eps (distância máxima para considerar vizinho)?
"""
outlier_detection = DBSCAN(eps=10000, min_samples=1)
db = outlier_detection.fit(X)

#############################
# 3 - Post processing
#############################
"""
Próximos passos: visualizar os clusteres por PCA
"""
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Updating flights df with cluster labels
prepared_data['cluster'] = labels

# Visualizing anomalies
anomalies = prepared_data[prepared_data['cluster'] != 0]
anomalous_flights = [flight[0] for flight in
                     anomalies['flight'].str.split('.')]

df_flights = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.dirname(currentdir)), 'local/compiled_flights.csv'
    )
)

df_flights_resampled = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.dirname(currentdir)), 'local/resampled_compiled_flights.csv'
    )
)

quantile_10 = get_quantile_per_sample(df_flights_resampled, 0.1,
                                          "DistanceFromTouchdown")
quantile_90 = get_quantile_per_sample(df_flights_resampled, 0.9,
                                          "DistanceFromTouchdown")

_ = plot_flights_n_boundary(df_flights,
                            'DistanceFromTouchdown', 'Altitude',
                            xlims=(-2, 1),
                            xlabel='Distance from touchdown [NM]',
                            ylabel='Altitude [ft]',
                            highlight_flights=anomalous_flights,
                            include_legend=True,
                            df_lower_boundary=quantile_10,
                            df_upper_boundary=quantile_90,
                            lower_boundary_label='10th percentile',
                            upper_boundary_label='90th percentile'
                            )
_ = plot_flights_n_boundary(df_flights,
                            'DistanceFromTouchdown', 'AirSpeed',
                            xlims=(-2, 1),
                            xlabel='Distance from touchdown [NM]',
                            ylabel='AirSpeed [m/s]',
                            highlight_flights=anomalous_flights,
                            include_legend=True,
                            df_lower_boundary=quantile_10,
                            df_upper_boundary=quantile_90,
                            lower_boundary_label='10th percentile',
                            upper_boundary_label='90th percentile'
                            )
_ = plot_flights_n_boundary(df_flights,
                            'DistanceFromTouchdown', 'Landing_Gear',
                            xlims=(-2, 1),
                            xlabel='Distance from touchdown [NM]',
                            ylabel='Landing_Gear',
                            highlight_flights=anomalous_flights,
                            include_legend=True,
                            df_lower_boundary=quantile_10,
                            df_upper_boundary=quantile_90,
                            lower_boundary_label='10th percentile',
                            upper_boundary_label='90th percentile'
                            )
_ = plot_flights_n_boundary(df_flights,
                            'DistanceFromTouchdown', 'Thrust_Rev',
                            xlims=(-2, 1),
                            xlabel='Distance from touchdown [NM]',
                            ylabel='Thrust_Rev',
                            highlight_flights=anomalous_flights,
                            include_legend=True,
                            df_lower_boundary=quantile_10,
                            df_upper_boundary=quantile_90,
                            lower_boundary_label='10th percentile',
                            upper_boundary_label='90th percentile'
                            )
