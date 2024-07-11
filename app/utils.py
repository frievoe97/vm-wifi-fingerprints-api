import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from schemas import RouterData

def process_received_data(routers: List[RouterData]):
    """
    Process received WiFi fingerprint data and sort it by MAC address.

    Parameters:
    routers (List[RouterData]): List of RouterData objects containing 'bssid' and 'signal_strength'.

    Returns:
    dict: A dictionary mapping 'bssid' to 'signal_strength'.
    """
    sorted_data = sorted(routers, key=lambda x: x.bssid)
    processed_data = {entry.bssid: entry.signal_strength for entry in sorted_data}
    return processed_data


def process_fingerprint_data(data):
    """
    Process fetched WiFi fingerprint data into a suitable format for modeling.

    Parameters:
    data (list): List of dictionaries containing 'measurement_id', 'room_id', 'signal_strength', 'ssid', and 'bssid'.

    Returns:
    dict: Nested dictionary where the keys are room IDs and measurement IDs,
          and the values are lists of fingerprints with 'mac_address', 'signal_strength', and 'ssid'.
    """
    rooms = {}
    for row in data:
        measurement_id = row['measurement_id']
        room_id = row['room_id']
        signal_strength = row['signal_strength']
        ssid = row['ssid']
        bssid = row['bssid']

        if room_id not in rooms:
            rooms[room_id] = {}

        if measurement_id not in rooms[room_id]:
            rooms[room_id][measurement_id] = []

        rooms[room_id][measurement_id].append(
            {'mac_address': bssid, 'signal_strength': signal_strength, 'ssid': ssid}
        )

    return rooms


def remove_non_eduroam_bssids(rooms):
    """
    Remove non-eduroam BSSIDs from the rooms data.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.

    Returns:
    dict: The filtered rooms dictionary with only 'eduroam' SSIDs.
    """
    for room_id in list(rooms.keys()):
        for measurement_id in list(rooms[room_id].keys()):
            rooms[room_id][measurement_id] = [
                router for router in rooms[room_id][measurement_id]
                if router['ssid'] in ['eduroam', 'HowToUseEduroam', 'Gast@HTW']
            ]
            if not rooms[room_id][measurement_id]:
                del rooms[room_id][measurement_id]
        if not rooms[room_id]:
            del rooms[room_id]
    return rooms


def remove_unreceived_bssids(rooms, received_data: List[RouterData]):
    """
    Remove all entries in rooms whose MAC address is not in received_data.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.
    received_data (dict): Dictionary where the keys are MAC addresses and the values are associated data (e.g., signal strength).

    Returns:
    dict: The filtered rooms dictionary where all fingerprints with MAC addresses not present in received_data have been removed.
    """
    received_mac_addresses = set(received_data.keys())

    for room_id in list(rooms.keys()):
        for measurement_id in list(rooms[room_id].keys()):
            filtered_fingerprint = [
                entry for entry in rooms[room_id][measurement_id]
                if entry['mac_address'] in received_mac_addresses
            ]

            if filtered_fingerprint:
                rooms[room_id][measurement_id] = filtered_fingerprint
            else:
                del rooms[room_id][measurement_id]  # Remove measurement if it has no valid fingerprints

        if not rooms[room_id]:  # Remove room if it has no measurements
            del rooms[room_id]

    return rooms


def calculate_average_signal_strength(rooms):
    """
    Calculate the average signal strength for each MAC address in each room.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.

    Returns:
    dict: Dictionary with averaged signal strengths for each room.
    """
    averaged_rooms = {}

    for room_id, measurements in rooms.items():
        mac_addresses = {}

        for measurement_id, fingerprints in measurements.items():
            for fingerprint in fingerprints:
                mac = fingerprint['mac_address']
                signal = fingerprint['signal_strength']
                if mac not in mac_addresses:
                    mac_addresses[mac] = []
                mac_addresses[mac].append(signal)

        averaged_fingerprints = []
        for mac, signals in mac_addresses.items():
            avg_signal = sum(signals) / len(signals)
            averaged_fingerprints.append({'mac_address': mac, 'signal_strength': avg_signal})

        averaged_rooms[room_id] = {1: averaged_fingerprints}

    return averaged_rooms


def calculate_executive_average(rooms, order=3):
    """
    Calculate the moving average of signal strengths for each room, using all measurements together.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.
    order (int): The order of the moving average to calculate. Default is 3. Can be set to 3 or 5.

    Returns:
    dict: A new dictionary with the same structure as the input 'rooms', where the signal strengths have been replaced by their calculated moving averages.
    """

    def moving_average(data, order):
        return pd.Series(data).rolling(window=order, min_periods=1).mean().tolist()

    for room_id, measurements in rooms.items():
        mac_addresses = {}
        for measurement_id, fingerprints in measurements.items():
            for fingerprint in fingerprints:
                mac = fingerprint['mac_address']
                signal = fingerprint['signal_strength']
                if mac not in mac_addresses:
                    mac_addresses[mac] = []
                mac_addresses[mac].append(signal)

        mac_averages = {}
        for mac, signals in mac_addresses.items():
            mac_averages[mac] = moving_average(signals, order)

        for measurement_id, fingerprints in measurements.items():
            for fingerprint in fingerprints:
                mac = fingerprint['mac_address']
                if mac in mac_averages and mac_averages[mac]:
                    fingerprint['signal_strength'] = mac_averages[mac].pop(0)

    return rooms


def handle_router_rssi_threshold(X, X_new, router_rssi_threshold=-100):
    """
    Handle router RSSI threshold by capping the signal strengths below the threshold.

    Parameters:
    X (numpy.ndarray): Training data matrix.
    X_new (numpy.ndarray): New data matrix.
    router_rssi_threshold (int): Threshold for router RSSI. Default is -100.

    Returns:
    tuple: Scaled training and new data matrices.
    """
    X_scaled = np.where(X < router_rssi_threshold, -100, X)
    X_new_scaled = np.where(X_new < router_rssi_threshold, -100, X_new)

    return X_scaled, X_new_scaled


def positive_values_representation(rssi_values, min_rssi):
    """
    Convert RSSI values to positive values.

    Parameters:
    rssi_values (numpy.ndarray): Array of RSSI values.
    min_rssi (int): Minimum RSSI value.

    Returns:
    numpy.ndarray: Positive RSSI values.
    """
    result = rssi_values - min_rssi
    return result.astype(np.float64)  # Ensure the correct type


def exponential_representation(rssi_values, min_rssi, alpha):
    """
    Apply exponential scaling to RSSI values.

    Parameters:
    rssi_values (numpy.ndarray): Array of RSSI values.
    min_rssi (int): Minimum RSSI value.
    alpha (float): Scaling factor.

    Returns:
    numpy.ndarray: Exponentially scaled RSSI values.
    """
    positive_values = positive_values_representation(rssi_values, min_rssi)
    divided_values = positive_values / alpha
    result = np.exp(divided_values)
    return result


def powed_representation(rssi_values, min_rssi, beta):
    """
    Apply power scaling to RSSI values.

    Parameters:
    rssi_values (numpy.ndarray): Array of RSSI values.
    min_rssi (int): Minimum RSSI value.
    beta (float): Power factor.

    Returns:
    numpy.ndarray: Power scaled RSSI values.
    """
    positive_values = positive_values_representation(rssi_values, min_rssi)
    result = (positive_values ** beta) / (abs(min_rssi) ** beta)
    return result


def normalize(values):
    """
    Normalize values to a range between 0 and 1.

    Parameters:
    values (numpy.ndarray): Array of values to normalize.

    Returns:
    numpy.ndarray: Normalized values.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)


def value_scaling(X, X_new, min_rssi_value=-100, value_scaling_strategy='none'):
    """
    Apply value scaling to the data matrices based on the chosen strategy.

    Parameters:
    X (numpy.ndarray): Training data matrix.
    X_new (numpy.ndarray): New data matrix.
    min_rssi_value (int): Minimum RSSI value. Default is -100.
    value_scaling_strategy (str): Strategy for scaling values. Must be 'none', 'exponential', 'powed', or 'positive'.

    Returns:
    tuple: Scaled training and new data matrices.
    """
    alpha = 24
    beta = np.e

    if value_scaling_strategy == 'exponential':
        X_scaled = exponential_representation(X, min_rssi_value, alpha)
        X_new_scaled = exponential_representation(X_new, min_rssi_value, alpha)
    elif value_scaling_strategy == 'powed':
        X_scaled = powed_representation(X, min_rssi_value, beta)
        X_new_scaled = powed_representation(X_new, min_rssi_value, beta)
    elif value_scaling_strategy == 'positive':
        X_scaled = positive_values_representation(X, min_rssi_value)
        X_new_scaled = positive_values_representation(X_new, min_rssi_value)
    elif value_scaling_strategy == 'none':
        X_scaled = X
        X_new_scaled = X_new
    else:
        raise ValueError("Invalid value_scaling_strategy. Must be 'none', 'exponential', 'powed', 'positive'")

    if value_scaling_strategy != 'none':
        X_scaled = normalize(X_scaled)
        X_new_scaled = normalize(X_new_scaled)

    return X_scaled, X_new_scaled


def remove_rare_routers(rooms, threshold):
    """
    Remove routers that appear less frequently than the given threshold.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.
    threshold (float): Minimum required count threshold for routers.

    Returns:
    dict: The filtered rooms dictionary with rare routers removed.
    """
    filtered_rooms = {}

    for room, measurements in rooms.items():
        total_measurements = len(measurements)

        router_counts = {}
        for measurement_id, values in measurements.items():
            for value in values:
                mac_address = value['mac_address']
                if mac_address not in router_counts:
                    router_counts[mac_address] = 0
                router_counts[mac_address] += 1

        min_required_count = threshold * total_measurements

        rare_routers = {mac for mac, count in router_counts.items() if count < min_required_count}

        filtered_measurements = {}
        for measurement_id, values in measurements.items():
            filtered_values = [value for value in values if value['mac_address'] not in rare_routers]
            if filtered_values:
                filtered_measurements[measurement_id] = filtered_values

        if filtered_measurements:
            filtered_rooms[room] = filtered_measurements

    return filtered_rooms


def handle_missing_values(X, mac_address_list, received_data, strategy='use_received'):
    """
    Handle missing values in the provided data matrix X.

    Parameters:
    X (numpy.ndarray): The data matrix with measurements for each room.
    mac_address_list (list): List of MAC addresses corresponding to the columns of X.
    received_data (dict): Dictionary with new measurement data.
    strategy (str): Strategy to handle missing values ('zero', '-100', 'use_received').

    Returns:
    numpy.ndarray: The updated data matrix X with missing values handled.
    """
    if strategy not in ['zero', '-100', 'use_received']:
        raise ValueError("Strategy must be one of 'zero', '-100', or 'use_received'")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] is None:
                if strategy == 'zero':
                    X[i, j] = 0
                elif strategy == '-100':
                    X[i, j] = -100
                elif strategy == 'use_received':
                    mac_address = mac_address_list[j]
                    X[i, j] = received_data.get(mac_address, 0)

    return X


def svm(X, X_new, y, kernel='rbf', C=1.0, gamma='scale'):
    """
    Perform SVM to find the nearest room based on received data.

    Parameters:
    X (numpy.ndarray): Training data matrix.
    X_new (numpy.ndarray): New data matrix.
    y (numpy.ndarray): Target labels.
    kernel (str): Kernel type to be used in the algorithm. Default is 'rbf'.
    C (float): Regularization parameter. Default is 1.0.
    gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.

    Returns:
    tuple: Predicted room, the decision function distance, and the used gamma value.
    """
    try:
        svm_model = SVC(kernel=kernel, C=C, probability=True, gamma=gamma)
        svm_model.fit(X, y)
        proba = svm_model.predict_proba(X_new)
        top_index = np.argmax(proba, axis=1)[0]
        distance = 1 - proba[0][top_index]
        used_gamma = svm_model._gamma
        print(f"Used gamma: {used_gamma}; Gamma Parameter: {gamma}")
        return svm_model.classes_[top_index], distance, used_gamma
    except Exception as e:
        print(f"Error: {e}")
        return []


def random_forest(X, X_new, y, n_estimators=100, max_depth=None):
    """
    Perform Random Forest to find the nearest room based on received data.

    Parameters:
    X (numpy.ndarray): Training data matrix.
    X_new (numpy.ndarray): New data matrix.
    y (numpy.ndarray): Target labels.
    n_estimators (int): The number of trees in the forest. Default is 100.

    Returns:
    tuple: Predicted room and the decision function distance.
    """

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    try:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf_model.fit(X, y)
        # logger.info(f"Number of features: {rf_model.n_features_}")
        proba = rf_model.predict_proba(X_new)
        top_index = np.argmax(proba, axis=1)[0]
        distance = proba[0][top_index]
        return rf_model.classes_[top_index], distance
    except Exception as e:
        print(f"Error: {e}")
        return []


def sorensen_distance(x, y):
    """
    Compute Sorensen distance between two arrays.

    Parameters:
    x (numpy.ndarray): First array.
    y (numpy.ndarray): Second array.

    Returns:
    float: Sorensen distance.
    """
    return np.sum(np.abs(x - y)) / np.sum(np.abs(x) + np.abs(y))


def euclidean_distance(x_value, y_value):
    """
    Compute Euclidean distance between two arrays.

    Parameters:
    x_value (numpy.ndarray): First array.
    y_value (numpy.ndarray): Second array.

    Returns:
    float: Euclidean distance.
    """
    return np.sqrt(np.sum((x_value - y_value) ** 2))


def knn(X, X_new, y, n_neighbors=10, metric='euclidean', weights='distance'):
    """
    Perform k-Nearest Neighbors to find the nearest room based on received data.

    Parameters:
    X (numpy.ndarray): Training data matrix.
    X_new (numpy.ndarray): New data matrix.
    y (numpy.ndarray): Target labels.
    n_neighbors (int): Number of neighbors to use. Default is 10.
    metric (str): Metric to use for distance computation. Default is 'euclidean'.
    weights (str): Weight function used in prediction. Default is 'distance'.

    Returns:
    tuple: Predicted room and the distance to the nearest neighbor.
    """
    try:
        if metric == 'sorensen':
            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=sorensen_distance, weights=weights)
        else:
            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=euclidean_distance, weights=weights)

        knn_model.fit(X, y)
        proba = knn_model.predict_proba(X_new)
        dist, indi = knn_model.kneighbors(X_new, n_neighbors=n_neighbors)
        top_indices = np.argsort(proba, axis=1)[:, -n_neighbors:][0][::-1]
        top_rooms = [(knn_model.classes_[i], proba[0][i]) for i in top_indices]
        return top_rooms[0][0], dist[0][0]
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def prepare_data(rooms):
    """
    Prepare the data from the rooms dictionary for kNN.

    Parameters:
    rooms (dict): Nested dictionary with room IDs and measurement IDs as keys, and lists of fingerprints as values.

    Returns:
    tuple: Feature matrix, target labels, and list of unique MAC addresses.
    """
    X = []
    y = []
    mac_address_list = []

    for room_id, measurements in rooms.items():
        for signals in measurements.values():
            for signal in signals:
                if signal['mac_address'] not in mac_address_list:
                    mac_address_list.append(signal['mac_address'])

    for room_id, measurements in rooms.items():
        for signals in measurements.values():
            features = [None] * len(mac_address_list)
            for signal in signals:
                index = mac_address_list.index(signal['mac_address'])
                features[index] = signal['signal_strength']
            X.append(features)
            y.append(room_id)

    return np.array(X), np.array(y), mac_address_list


def prepare_received_data(received_data, mac_address_list):
    """
    Prepare the received data in the same format as the training data.

    Parameters:
    received_data (dict): Dictionary with received data where keys are MAC addresses and values are signal strengths.
    mac_address_list (list): List of MAC addresses.

    Returns:
    numpy.ndarray: Feature vector for the received data.
    """
    features = [-100] * len(mac_address_list)
    for mac_address, signal_strength in received_data.items():
        if mac_address in mac_address_list:
            index = mac_address_list.index(mac_address)
            features[index] = signal_strength
    return np.array([features])
