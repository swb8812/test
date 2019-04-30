import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

def get_data(input_features):
    """Get data from raw text files.
    Args:
        input_features: List[str], feature names.
            Method finds files by formatting feature names as "./FEATURE_NAME.txt".
            And get last element for each features, and second element as target values.
    Returns:
        pandas.DataFrame, data read from text files.
        Column: each features including target values
        Row: each data
    """
    data = {}
    def getter(line, n): 
        return float(line.strip().split('\t')[n])

    def formatter(name): 
        return './{}.txt'.format(name)

    with open(formatter(input_features[0])) as f:
        data['target'] = []
        for line in f:
            data['target'].append(getter(line, 1))

    for name in input_features:
        with open(formatter(name)) as f:
            data[name] = []
            for line in f:
                data[name].append(getter(line, -1))
    return pd.DataFrame(data)


def rmse(a, b):
    """Return Root Mean Sqaured Error"""
    return np.sqrt(((a-b)**2).mean())


def main(_):
    input_features = ['holiday', 'starttime', 'weekday', 'humidity', 'rank']

    # Define model. It is fine tuned model, and reference is on the link of google drive.
    model = MLPRegressor(hidden_layer_sizes=[256, 1],
                         activation='relu',
                         solver='adam',
                         alpha=0,
                         batch_size=128,
                         learning_rate_init=0.01,
                         max_iter=10000,
                         random_state=1024)
    
    # Read data.
    data = get_data(input_features)
    # Shuffling it.
    np.random.shuffle(data.values)

    # Divide into test and training set.
    train_rate = 0.9
    train_set = data.loc[:train_rate * len(data)]
    test_set = data.loc[train_rate * len(data):]

    train_x = train_set[input_features]
    train_y = train_set['target']

    test_x = test_set[input_features]
    test_y = test_set['target']

    # Fitting model.
    model.fit(train_x, train_y)

    # Print RMSE.
    print('train rmse: ' + str(rmse(train_y, model.predict(train_x))))
    print('test rmse: ' + str(rmse(test_y, model.predict(test_x))))

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
