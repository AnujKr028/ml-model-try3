# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import networkx as nx

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model, graph, and label encoder
model = joblib.load("model.pkl")
# graph = joblib.load("graph.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# Load the scaler
scaler = StandardScaler()

# Define a function to preprocess input data
def preprocess_input(data):
    """
    Preprocesses the input data for prediction.
    """
    # Convert categorical features to numeric using the label encoder
    data['start'] = label_encoder.transform([str(data['start'])])[0]
    data['end'] = label_encoder.transform([str(data['end'])])[0]

    # Select relevant features
    features = ['Time', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'start', 'end']
    input_data = pd.DataFrame([data])[features]

    # Scale the data
    scaled_data = scaler.fit_transform(input_data)
    return scaled_data

@app.route('/predict_traffic', methods=['POST'])
def predict_traffic():
    """
    Predicts the traffic situation based on input features.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Make predictions
        prediction = model.predict(processed_data)[0]

        # Map prediction to human-readable labels
        traffic_labels = {0: 'low', 1: 'normal', 2: 'high', 3: 'heavy'}
        result = traffic_labels.get(prediction, "Unknown")

        return jsonify({"traffic_situation": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/shortest_path', methods=['GET'])
def shortest_path():
    """
    Finds the shortest path between two locations in the graph.
    """
    try:
        # Get source and destination from query parameters
        source = int(request.args.get('source'))
        destination = int(request.args.get('destination'))

        # Find the shortest path using Dijkstra's algorithm
        shortest_path = nx.shortest_path(graph, source=source, target=destination, weight='weight')
        shortest_distance = nx.shortest_path_length(graph, source=source, target=destination, weight='weight')

        # Decode the path using the label encoder
        decoded_path = [label_encoder.inverse_transform([node])[0] for node in shortest_path]

        return jsonify({
            "shortest_path": decoded_path,
            "total_distance": shortest_distance
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
