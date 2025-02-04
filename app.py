import os
import pickle
from datetime import datetime
from io import BytesIO
import base64

import matplotlib
matplotlib.use('Agg')  # Fix for non-GUI environments

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Secret key for session management
app.secret_key = 'your_secret_key'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# User Model for Signup/Login
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # Admin flag for user

# History Model to store user predictions
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    restaurant = db.Column(db.String(100), nullable=False)  # Added restaurant field
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp for predictions
    user = db.relationship('User', backref=db.backref('history', lazy=True))

# Load the trained machine learning model
model_path = os.path.join("model", "food_order_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists. Try another."

        # Add new user to the database
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Authenticate user
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid username or password. Try again."

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction_text = None
    restaurants = ['Pizza Place', 'Burger Shack', 'Sushi Spot', 'Italian Bistro', 'Mexican Grill']  # Random restaurant names

    if request.method == 'POST':
        try:
            # Get input values from form
            age = request.form['age']
            marital_status = request.form['marital_status']
            occupation = request.form['occupation']
            income = request.form['income']
            education = request.form['education']
            family_size = request.form['family_size']
            feedback = request.form['feedback']
            restaurant = request.form['restaurant']  # Get selected restaurant

            # Check if any field is empty
            if not all([age, marital_status, occupation, income, education, family_size, feedback, restaurant]):
                prediction_text = "All fields are required. Please fill out the form completely."
                return render_template('predict.html', prediction=prediction_text, restaurants=restaurants)

            # Convert numeric values to integers (with exception handling for invalid input)
            try:
                age = int(age)
                marital_status = int(marital_status)
                income = int(income)
                family_size = int(family_size)
                feedback = int(feedback)
            except ValueError:
                prediction_text = "Invalid input, please ensure all fields are filled correctly with valid numbers."
                return render_template('predict.html', prediction=prediction_text, restaurants=restaurants)

            # Mapping for occupation and education (update as per your data)
            occupation_mapping = {
                'Engineer': 0,
                'Doctor': 1,
                'Teacher': 2,
                'Artist': 3,
                'Others': 4
            }
            education_mapping = {
                'High School': 0,
                'Graduate': 1,
                'Postgraduate': 2,
                'Others': 3
            }

            # Validate occupation and education inputs
            if occupation not in occupation_mapping:
                prediction_text = "Invalid occupation. Please select a valid option."
                return render_template('predict.html', prediction=prediction_text, restaurants=restaurants)

            if education not in education_mapping:
                prediction_text = "Invalid education level. Please select a valid option."
                return render_template('predict.html', prediction=prediction_text, restaurants=restaurants)

            # Convert string inputs to numerical values
            occupation = occupation_mapping[occupation]
            education = education_mapping[education]

            # If feedback is negative (0 or low value), force "Will Not Order Again" prediction
            if feedback <= 2:  # Adjust the threshold based on your feedback scale
                prediction_text = "Will Not Order Again"
            else:
                # Prepare input data for model
                input_data = [[age, marital_status, occupation, income, education, family_size, feedback]]

                # Predict using the trained model
                prediction = model.predict(input_data)[0]
                prediction_text = "Will Order Again" if prediction == 1 else "Will Not Order Again"

            # Save prediction to the history with selected restaurant
            user = User.query.filter_by(username=session['user']).first()
            new_history = History(user_id=user.id, prediction=prediction_text, restaurant=restaurant)
            db.session.add(new_history)
            db.session.commit()

        except Exception as e:
            prediction_text = f"An error occurred: {str(e)}"

    return render_template('predict.html', prediction=prediction_text, restaurants=restaurants)

# Dashboard Route with Bar Chart
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['user']).first()
    history = History.query.filter_by(user_id=user.id).all()

    # Generate a bar chart for user predictions
    if not history:
        return render_template('dashboard.html', history=history, image_base64=None, message="No predictions yet.")

    predictions = [record.prediction for record in history]
    prediction_counts = pd.Series(predictions).value_counts()

    # Create the bar chart
    plt.figure(figsize=(6, 4))
    prediction_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Your Prediction Summary")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    # Save chart as image to BytesIO
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches="tight")
    plt.close()
    img_stream.seek(0)

    # Convert the image to a base64 string
    img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

    return render_template('dashboard.html', history=history, image_base64=img_base64)

# Generate PDF Report with Chart and Details
@app.route('/generate_report')
def generate_report():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['user']).first()
    history = History.query.filter_by(user_id=user.id).all()

    if not history:
        return "No prediction history available."

    # Count predictions
    predictions = [record.prediction for record in history]
    prediction_counts = pd.Series(predictions).value_counts()

    # Breakdown of predictions (e.g., Will Order Again vs. Will Not Order Again)
    will_order_again_count = prediction_counts.get("Will Order Again", 0)
    will_not_order_again_count = prediction_counts.get("Will Not Order Again", 0)

    # Generate bar chart
    plt.figure(figsize=(6, 4))
    prediction_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Prediction Summary")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    # Save chart to BytesIO
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches="tight")
    plt.close()
    img_stream.seek(0)

    # Convert to PIL Image and save as temporary file
    img = Image.open(img_stream)
    temp_img_path = "temp_chart.png"
    img.save(temp_img_path, format='PNG')

    # Create PDF Report
    pdf_stream = BytesIO()
    c = canvas.Canvas(pdf_stream, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Add unique details about the user
    c.drawString(100, 750, f"Prediction Report for {user.username}")
    c.drawString(100, 730, "Prediction History Summary")

    # Total number of predictions and breakdown
    c.drawString(100, 710, f"Total Predictions: {len(history)}")
    c.drawString(100, 690, f"Predicted 'Will Order Again': {will_order_again_count}")
    c.drawString(100, 670, f"Predicted 'Will Not Order Again': {will_not_order_again_count}")

    # Include details of each prediction (optional: limit to the most recent 5 or 10 predictions for brevity)
    y_position = 650
    for record in history[:10]:  # Display top 10 recent predictions
        c.drawString(100, y_position, f"{record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {record.prediction} (Restaurant: {record.restaurant})")
        y_position -= 20
        if y_position < 100:  # Prevent content overflow
            break

    # Attach chart to PDF
    c.drawImage(temp_img_path, 100, 300, width=400, height=200)
    c.save()

    pdf_stream.seek(0)
    return send_file(pdf_stream, as_attachment=True, download_name=f"{user.username}_prediction_report.pdf", mimetype="application/pdf")

# Main entry point
if __name__ == "__main__":
    app.run(debug=True)
