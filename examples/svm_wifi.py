import polymath as pm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split



def get_data():
    rssi_datafile = 'data/measure1_smartphone_wifi.csv'
    rssi_data = pd.read_csv(rssi_datafile, header=None)

    smartphone_datafile = 'data/measure1_smartphone_sens.csv'
    smartphone_data = pd.read_csv(smartphone_datafile)

    timestamp_datafile = 'data/measure1_timestamp_id.csv'
    timestamp_data = pd.read_csv(timestamp_datafile, header=None)

    points_mapping_file = 'data/PointsMapping.csv'
    points_mapping_data = pd.read_csv(points_mapping_file)
    points_mapping_data = points_mapping_data[['ID', 'X', 'Y']]
    points_mapping_data.ID.astype({'ID': int})
    points_mapping_data['ID'] = points_mapping_data['ID'] - 1

    return rssi_data, smartphone_data, timestamp_data, points_mapping_data


def preprocess_data(rssi_data, smartphone_data, timestamp_data, points_mapping_data):
    rssi_column_headers = [f'RSSI_{i}' for i in range(127)]
    rssi_column_headers.insert(0, 'ID')
    rssi_data.columns = rssi_column_headers
    rssi_merged = points_mapping_data.merge(rssi_data, on='ID')
    return rssi_merged


def merge_data(rssi_merged, smartphone_data, timestamp_data, points_mapping_data):
    # Add RSSI columns from 1 to 127 to smartphone geomagnetic dataset and set 0 as default vaule
    for i in range(127):
        smartphone_data[f'RSSI_{i + 1}'] = 0

    #print(rssi_merged)
    for index, timestamp in timestamp_data.iterrows():
        ts_from = timestamp[0]
        ts_to = timestamp[1]
        placeid = timestamp[2]
        #print(placeid)
        rssi = rssi_merged.iloc[placeid][3:]
        coordinates = rssi_merged.iloc[placeid][1:3]

        for c in range(127):
            smartphone_data.loc[smartphone_data['timestamp'].between(ts_from, ts_to), f'RSSI_{c + 1}'] = rssi[c]
        smartphone_data.loc[smartphone_data['timestamp'].between(ts_from, ts_to), 'PlaceID'] = placeid
        smartphone_data.loc[smartphone_data['timestamp'].between(ts_from, ts_to), 'X'] = coordinates[0]
        smartphone_data.loc[smartphone_data['timestamp'].between(ts_from, ts_to), 'Y'] = coordinates[1]

    smartphone_data = smartphone_data[smartphone_data['PlaceID'] > -1]


def split_data(smartphone_data):
    X = smartphone_data.iloc[:, 1:140]
    y = smartphone_data.iloc[:, 140]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def np_reference(x_train, y_train, weights, lr=0.00001, delta=10):
    output_info = {}
    learning_rate = lr
    delta = delta
    dW = np.zeros(weights.shape)

    scores = np.dot(x_train, weights)
    output_info["scores"] = scores

    correct_class_score = scores[y_train]
    output_info["correct_class_score"] = correct_class_score

    h = (scores - correct_class_score + delta)
    output_info["h"] = h

    margin = np.greater(h, 0).astype(float)
    margin[y_train] = 0
    output_info["margin"] = margin

    valid_margin_count = margin.sum()
    output_info["valid_margin_count"] = valid_margin_count

    updated_margin = margin.copy()
    updated_margin[y_train] -= valid_margin_count
    output_info["updated_margin"] = updated_margin

    x = x_train.reshape(x_train.shape[0], 1)
    updated_margin = updated_margin.reshape(1, updated_margin.shape[0])

    dW = np.dot(x, updated_margin)
    output_info["dW"] = dW

    weights = weights - learning_rate * dW
    output_info["weights"] = weights

    return output_info

def generate_polymath(lr, delta, features=127, locations=325, train_size=7703):
    with pm.Node(name="svm") as graph:
        learning_rate = pm.parameter("learning_rate", default=lr)
        delta = pm.parameter("delta", default=delta)
        n_features = pm.parameter("n_features", default=features)
        n_locations = pm.parameter("n_locations", default=locations)
        x_train = pm.input("x_train", shape=(n_features))
        y_train = pm.input("y_train", shape=(n_locations,))
        y_train_inv = pm.input("y_train_inv", shape=(n_locations,))
        weights = pm.state("weights", shape=(n_features, n_locations))

        i = pm.index(0, n_features - 1, name="i")
        j = pm.index(0, n_locations - 1, name="j")
        #
        scores = pm.sum([i], (weights[i, j] * x_train[i]), name="scores")
        correct_class_score = pm.sum([j], (scores[j] * y_train[j]), name="correct_class_score")
        #
        h = ((scores[j] - correct_class_score + delta).set_name("h") > 0)
        margin = (pm.cast(np.float32, h[j]) * y_train_inv[j]).set_name("margin")
        valid_margin_count = pm.sum([j], margin[j], name="valid_margin_count")
        partial = (y_train[j]*valid_margin_count).set_name("partial")
        updated_margin = (margin[j] - partial[j]).set_name("updated_margin")
        # #
        dW = (x_train[i]*updated_margin[j]).set_name("dW")
        weights[i, j] = weights[i, j] - learning_rate * dW[i, j]

    return graph

def generate_data(n_features=127, n_locations=325):
    rssi_data, smartphone_data, timestamp_data, points_mapping_data = get_data()
    rssi_data = preprocess_data(rssi_data, smartphone_data, timestamp_data, points_mapping_data)
    merge_data(rssi_data, smartphone_data, timestamp_data, points_mapping_data)
    X_train, X_test, y_train, y_test = split_data(smartphone_data)
    input_info = {}
    input_info["x_train"] = X_train
    input_info["weights"] = np.random.randint(-3, 50, n_features * n_locations).reshape(n_features, n_locations)
    input_info["y_train"] = y_train
    input_info["learning_rate"] = 0.00001
    return X_train, X_test, y_train, y_test, input_info

def get_sample_values(n_features, n_locations, lr, delta):
    input_info = {}
    input_info["x_train"] = np.random.randint(-3, 50, n_features)
    input_info["weights"] = np.random.randint(-3, 50, n_features * n_locations).reshape(n_features, n_locations)
    input_info["y_train_np"] = np.random.randint(0, n_locations, 1)[0]

    input_info["y_train"] = np.zeros(n_locations, dtype=np.int)
    input_info["y_train"][input_info["y_train_np"]] = 1
    input_info["y_train_inv"] = np.ones(n_locations, dtype=np.int)
    input_info["y_train_inv"][input_info["y_train_np"]] = 0

    input_info["learning_rate"] = lr
    input_info["delta"] = delta
    # input_info["n_features"] = n_features
    # input_info["n_locations"] = n_locations
    return input_info

def main(store_graph=True, evaluate_samples=False, load_data=True, lr=0.1, delta=3, n_features=127, n_locations=325, train_size=7703):
    if load_data:
        X_train, X_test, y_train, y_test, input_info = generate_data()
    else:
        input_info = {}

    graph = generate_polymath(lr, delta, features=n_features, locations=n_locations, train_size=train_size)
    if evaluate_samples:
        key = "weights"
        sample_dict = get_sample_values(n_features, n_locations, lr, delta)
        np_res = np_reference(sample_dict["x_train"], sample_dict["y_train_np"],
                              sample_dict["weights"], lr, delta)
        sample_dict.pop("y_train_np")
        pm_res = graph(key, sample_dict)
        np.testing.assert_allclose(pm_res, np_res[key])

    if store_graph:
        shape_dict = {"n_features": n_features, "n_locations": n_locations}
        tabla_path = f"{graph.name}_tabla.json"
        tabla_ir, tabla_graph = pm.generate_tabla(graph,
                                                  shape_dict,
                                                  tabla_path,
                                                  context_dict=input_info,
                                                  add_kwargs=True,
                                                  debug=False)



if __name__ == '__main__':
    main(store_graph=True, evaluate_samples=False, load_data=False, n_features=20, n_locations=50)