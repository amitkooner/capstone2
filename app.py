from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model (Ensure the model file is in the same directory as this script or provide the correct path)
model = joblib.load('best_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert data into DataFrame (Ensure the input data aligns with the model's feature requirements)
    data = pd.DataFrame(data)
    
    # Predictions
    predictions = model.predict(data)
    
    # Convert predictions to list (for JSON serialization)
    predictions = predictions.tolist()
    
    # Send predictions back to the client
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
