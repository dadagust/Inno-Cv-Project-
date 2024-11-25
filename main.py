import os
from data_preparation import prepare_data
from train_model import train_the_model

if __name__ == '__main__':
    """
        If you want to train a specific dataset,
        you need to change the class mapping in train_model.py
    """
    if os.path.exists("data.csv"):
        train_the_model()
    else:
        prepare_data()
        train_the_model()
