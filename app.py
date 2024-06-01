import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model pipeline
with open('model/logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Extract features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        print("Input DataFrame: ", input_data)
        
        # Predict using the loaded model pipeline
        classification = model.predict(input_data)[0]
        return render_template('index.html', classification_text=f'Species: {classification}')
    except KeyError as e:
        return f"Missing form data: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)