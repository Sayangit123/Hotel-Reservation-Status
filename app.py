from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model_pipeline = joblib.load('model_pipeline.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        market_segment = str(request.form['market_segment'])
        distribution_channel = str(request.form['distribution_channel'])
        reserved_room_type = str(request.form['reserved_room_type'])
        assigned_room_type = str(request.form['assigned_room_type'])
        customer_type = str(request.form['customer_type'])
        reservation_status = str(request.form['reservation_status'])
        lead_time = int(request.form['lead_time'])
        arrival_date_year = int(request.form['arrival_date_year'])
        arrival_date_week_number = int(request.form['arrival_date_week_number'])
        arrival_date_day_of_month = int(request.form['arrival_date_day_of_month'])
        adults = int(request.form['adults'])
       
        # Create DataFrame from input features
        features = pd.DataFrame({
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'reserved_room_type': [reserved_room_type],
            'assigned_room_type': [assigned_room_type],
            'customer_type': [customer_type],
            'reservation_status': [reservation_status],
            'lead_time': [lead_time],
            'arrival_date_year': [arrival_date_year],
            'arrival_date_week_number': [arrival_date_week_number],
            'arrival_date_day_of_month': [arrival_date_day_of_month],
            'adults': [adults],
        })

         # Make prediction
        is_canceled = model_pipeline.predict(features)

        return render_template('index.html', prediction_text='Predicted is_canceled:{}'.format(is_canceled[0]))

if __name__ == "__main__":
    app.run(debug=True)

