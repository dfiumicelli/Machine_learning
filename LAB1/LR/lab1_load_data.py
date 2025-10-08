import pandas as pd

def load_housing_data(filename):

    data = pd.read_csv(filename)
    data.columns = ['sqft', 'bedrooms', 'price']
    return data[['sqft', 'bedrooms']], data['price']

if __name__ == '__main__':

    filename = 'datasets/ex1data2.txt'

    X, y = load_housing_data(filename)
    print(X)
