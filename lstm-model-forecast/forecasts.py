import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler


class Forecast:
    def __init__(self, df: DataFrame, product: str, obs: int, quantity: int):
        # load model
        self.df = df
        self.product = product
        self.quantity = quantity
        self.model = keras.models.load_model(f"forecast/models/model_product_{product}.h5")
        scaler = MinMaxScaler(feature_range=(0, 1))  # scale data

        for _ in range(self.quantity):
            dataset = self.df[-obs:].values  # get last obs to predict
            scaled = scaler.fit_transform(dataset)
            x_pred = np.array([scaled[:, 0]])  # obs as numpy array
            x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))  # Reshape the data

            # forecast values
            y_pred = self.model.predict(x_pred)
            y_pred = scaler.inverse_transform(y_pred)[:, 0]

            new_index = self.df.shift(
                1, freq="infer"
            ).last_valid_index()  # increment index by its frequency
            self.df.loc[new_index] = y_pred  # add to dataframe

    def print_plot_predictions(self):
        print(
            f"Product {self.product.upper()} - Forecast for the next 4 periods: {self.df[-self.quantity:].values[:, 0]}"
        )

        data = self.df[: -self.quantity]
        pred = self.df[-self.quantity - 1 :]

        # Plot the predictions
        plt.title(f"Producto {self.product.upper()}")
        plt.xlabel("Fecha")
        plt.ylabel("Venta (Unidades)")
        plt.plot(data["Venta\n(unidades)"], label="Data")
        plt.plot(pred["Venta\n(unidades)"], label="Predictions")
        plt.legend(loc=0)
        plt.show()


# SET DATA AS DF
def get_data(sheet_name: str):
    df = pd.read_excel("forecast/forecast-data.xlsx", sheet_name=sheet_name)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["Fecha"]))
    df.drop(columns=["Fecha"], inplace=True)

    return df.iloc[:, :1]


#  LOAD DATA
df_a = get_data("Producto A")
df_b = get_data("Producto B")
df_c = get_data("Producto C")


# GET FORECAST
# For the next 4 months
product_a = Forecast(df_a, product="a", obs=6, quantity=4)
product_a.print_plot_predictions()

# For the next 4 months
product_b = Forecast(df_b, product="b", obs=2, quantity=4)
product_b.print_plot_predictions()

# For the next 4 days
product_c = Forecast(df_c, product="c", obs=5, quantity=4)
product_c.print_plot_predictions()
