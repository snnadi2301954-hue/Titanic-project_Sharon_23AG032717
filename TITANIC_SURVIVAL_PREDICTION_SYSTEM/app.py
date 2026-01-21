from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model/titanic_survival_model.pkl", "rb") as file:
    model, scaler, le_sex, le_embarked = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = le_sex.transform([request.form["sex"]])[0]
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = le_embarked.transform([request.form["embarked"]])[0]

        data = np.array([[pclass, sex, age, fare, embarked]])
        data = scaler.transform(data)

        result = model.predict(data)[0]
        prediction = "Survived" if result == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
