import pandas as pd
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
import numpy as np

remove_nans = lambda x: x[~np.isnan(x)]

data_points_per_month = 2146

def read_csv(filename):
    return pd.read_csv(filename)

def build_dataframe(file_list):
    combined_df = pd.DataFrame()
    with Pool(processes=8) as pool:
        df_list = pool.map(read_csv, file_list)
        combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def arima(data, lag, d, q):
    data = remove_nans(data)
    model = ARIMA(data, order=(lag, d, q))
    fit = model.fit()
    preds = fit.predict(1, 7751, typ='levels')
    return preds

def plot(x_data, y_data, x_label, y_label, title, color):
    xi = list(range(len(x_data)))
    plt.plot(xi, y_data, color="blue")
    plt.xticks(xi, x_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.locator_params(axis="x", nbins=5)
    plt.show()

def weather_arima():
    weather_aug_to_sep = glob.glob("./weather-aug-sep-204/*.csv")
    weather_data = build_dataframe(weather_aug_to_sep).fillna(method='ffill')
    weather_data["date"] = pd.to_datetime(weather_data.date)
    weather_data.sort_values(by="date")

    predicted_df = pd.DataFrame()
    predicted_df["date"] = pd.date_range(start='2014-10-01',
         end='2014-10-30', periods=data_points_per_month)
    predicted_df["date"] = pd.to_datetime(predicted_df.date)

    weather_data["pressure"] = weather_data["pressure"].rolling(20).mean()
    predicted_df["pressure"] = arima(weather_data["pressure"].values, 10, 1, 1)

    weather_data["temperature"] = weather_data["temperature"].rolling(20).mean()
    predicted_df["temperature"] = arima(weather_data["temperature"].values, 2, 1, 1)

    weather_data["wspdm"] = weather_data["wspdm"].rolling(20).mean()
    predicted_df["wspdm"] = arima(weather_data["wspdm"].values, 2, 1, 1)

    predicted_df.to_csv("predicted-weather.csv")

    time_range = weather_data["date"].append(predicted_df["date"], ignore_index = True)
    pressures = weather_data["pressure"].append(predicted_df["pressure"], ignore_index = True)
    temperatures = weather_data["temperature"].append(predicted_df["temperature"], ignore_index = True)
    wind_speeds = weather_data["wspdm"].append(predicted_df["wspdm"], ignore_index = True)

    plot(time_range.values, pressures.values, "dates", "air pressure", "Air Pressure With Prediction", "black")
    plot(time_range.values, temperatures.values, "dates", "temperatures", "Temperature With Prediction", "red")
    plot(time_range.values, wind_speeds.values, "dates", "wind speed", "Wind Speed With Prediction", "blue")

def traffic_arima():
    traffic_aug_sept = glob.glob("./traffic-aug-sep-2014/trafficData158324.csv")
    traffic_data = build_dataframe(traffic_aug_sept).fillna(method='ffill')
    traffic_data["TIMESTAMP"] = pd.to_datetime(traffic_data.TIMESTAMP)
    traffic_data.sort_values(by="TIMESTAMP")

    predicted_df = pd.DataFrame()
    predicted_df["TIMESTAMP"] = pd.date_range(start='2014-10-01',
         end='2014-10-30', periods=7751)
    predicted_df["TIMESTAMP"] = pd.to_datetime(predicted_df.TIMESTAMP)

    traffic_data["avgSpeed"] = traffic_data["avgSpeed"].rolling(80).mean()
    plot(traffic_data["TIMESTAMP"], traffic_data["avgSpeed"], "timestamp", "average speed", "Average Speed", "blue")

    traffic_data["vehicleCount"] = traffic_data["vehicleCount"].rolling(80).mean()
    plot(traffic_data["TIMESTAMP"], traffic_data["vehicleCount"], "timestamp", "vehicle count", "Vehicle Count", "blue")

    predicted_df["avgSpeed"] = arima(traffic_data["avgSpeed"].values, 5, 1, 1)
    predicted_df["vehicleCount"] = arima(traffic_data["vehicleCount"].values, 5, 1, 1)

    predicted_df.to_csv("predicted-traffic.csv")

    time_range = traffic_data["TIMESTAMP"].append(predicted_df["TIMESTAMP"], ignore_index = True)
    avg_speeds = traffic_data["avgSpeed"].append(predicted_df["avgSpeed"], ignore_index = True)
    vehicle_counts = traffic_data["vehicleCount"].append(predicted_df["vehicleCount"], ignore_index = True)

    plot(time_range.values, avg_speeds.values, "timestamp", "avg speed", "Average Speed With Prediction", "blue")
    plot(time_range.values, vehicle_counts.values, "timestamp", "vehicle count", "Vehicle Count With Prediction", "blue")

def main():
    weather_arima()
    traffic_arima()

if __name__ == "__main__":
    main()
