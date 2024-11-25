import os
from data_preparation import prepare_data
from train_model import train_the_model
from train_canny_model import train_canny


if __name__ == '__main__':
    """
        If you want to train a model on a specific dataset,
        you need to change the class mapping in train_model.py/train_canny_model.py
    """
    canny = True
    if canny:
        num_epochs = 10
        train_canny(num_epochs)
    else:
        if os.path.exists("data.csv"):
            train_the_model()
        else:
            prepare_data()
            train_the_model()
