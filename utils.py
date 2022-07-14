import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ?utility function to split dataset into train and test
def split_data(X, y, test_size=0.2, random_state=42):
    """
     ? It will split the data into 20% for testing and rest as training.
      param: X - features of the data.
      param: y - target variable.
      param: test_size - test set size. Default is 20%.
      param: random_state - Default is 42.
      return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test