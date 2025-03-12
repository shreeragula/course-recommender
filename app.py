from flask import Flask, render_template, request
import pandas as pd
from ml_model import recommend_course  # Import your ML function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        course_name = request.form["course_name"]
        recommendations = recommend_course(course_name)  # Call ML function
    
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
