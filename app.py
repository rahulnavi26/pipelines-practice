from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        yr_built = float(request.form['yr_built'])

        arr = np.array([[bedrooms, bathrooms, floors, yr_built]])

        pred = model.predict(arr)

        prediction = float(pred[0][0])

        return render_template(
            'index.html',
            data=round(prediction, 2)
        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)