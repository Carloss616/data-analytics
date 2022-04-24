import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Activation, Dense
from keras.models import Sequential
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# CLASS FORECAST
class Training:
    def __init__(
        self,
        df: DataFrame,
        product: str,
        obs: int,
        percent_training: float = 0.7,
        nn_lstm: int = 15,
        nn_hidden: list[int] = [],
        linear_activation: bool = False,
        epochs: int = 950,
        batch_size: int = 32,
    ):
        self.df = df
        self.product = product
        # Convert the data frame to a numpy array
        self.dataset = df.values
        # Get the number of rows to train the model on
        self.training_data_len = math.ceil(len(self.dataset) * percent_training)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.dataset)

        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data[0 : self.training_data_len, :]
        # Split the data into x_train and y_train data sets
        self.x_train = []
        self.y_train = []

        # obs: Number of observations to predict
        for i in range(obs, len(train_data)):
            self.x_train.append(train_data[i - obs : i, 0])
            self.y_train.append(train_data[i, 0])

        # Convert the x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

        # Reshape the data
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(nn_lstm, input_shape=(self.x_train.shape[1], 1)))  # input
        # model.add(LSTM(50, return_sequences=False))
        for nn in nn_hidden:
            self.model.add(Dense(nn))  # hidden layer
        self.model.add(Dense(1))  # output
        if linear_activation:
            self.model.add(Activation("linear"))

        # Compile the model
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train, epochs=epochs, batch_size=batch_size
        )
        self.model.save(f"forecast/models/model_product_{self.product}.h5")

        # Create the testing data set
        # Create the scaled testing data set
        test_data = scaled_data[self.training_data_len - obs :, :]
        # Create the data sets x_test and y_test
        self.x_test = []
        for i in range(obs, len(test_data)):
            self.x_test.append(test_data[i - obs : i, 0])

        # Convert the data to a numpy array
        self.x_test = np.array(self.x_test)

        # Reshape the data
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

        # Get the models predicted price values
        self.y_test = self.model.predict(self.x_test)
        self.y_test = scaler.inverse_transform(self.y_test)

        # Get the root mean squared error (RMSE)
        self.y_true = self.dataset[self.training_data_len :, :]  # y_true
        self.rmse = np.sqrt(mean_squared_error(self.y_true, self.y_test))

    def plot_history(self):
        # Plot the training and validation loss for each epoch
        plt.plot(self.history.history["loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def get_comparisson(self):
        # valid = self.df.loc[self.training_data_len : , :]
        valid = self.df[self.training_data_len :].copy(deep=True)
        valid["Predictions"] = self.y_test
        return valid

    def plot_predictions(self):
        train = self.df[: self.training_data_len]
        valid = self.get_comparisson()

        # Plot the predictions
        plt.title(f"Producto {self.product.upper()}")
        plt.xlabel("Fecha")
        plt.ylabel("Venta (Unidades)")
        plt.plot(train["Venta\n(unidades)"], label="Train")
        plt.plot(valid["Venta\n(unidades)"], label="Real")
        plt.plot(valid["Predictions"], label="Predictions")
        plt.legend(loc=4)
        plt.show()

    def print_results(self):
        print(f"Root Mean Squared Error: {self.rmse:.2f}")
        print()
        print(self.get_comparisson().head())

    def print_plot_all(self):
        self.print_results()
        self.plot_predictions()
        self.plot_history()


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

print(df_a.head())
print(df_b.head())
print(df_c.head())

# GET AND SAVE MODEL PRODUCT A
product_a = Training(df_a, product="a", obs=6, linear_activation=True)
product_a.print_plot_all()

# GET AND SAVE MODEL PRODUCT B
product_b = Training(df_b, product="b", obs=2, batch_size=1, nn_hidden=[50, 50, 50, 50, 50])
product_b.print_plot_all()

# GET AND SAVE MODEL PRODUCT C
product_c = Training(df_c, product="c", obs=5, epochs=1000, batch_size=50, linear_activation=True)
product_c.print_plot_all()
