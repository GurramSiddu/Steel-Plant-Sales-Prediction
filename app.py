import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib

app = Flask(__name__)

# Load XGBoost models
sales_model_customer = joblib.load('sales_model_customer.xgb')
qty_model_customer = joblib.load('qty_model_customer.xgb')
sales_model_material = joblib.load('sales_model_material.xgb')
qty_model_material = joblib.load('qty_model_material.xgb')

# Directory to save the plot images
plot_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(plot_dir, exist_ok=True)

def predict_sales_and_qty(model_sales, model_qty, code_name, code, year, months):
    # Create DataFrame for prediction
    prediction_df = pd.DataFrame({
        code_name: [code] * len(months),
        'YEAR': [year + (month - 1) // 12 for month in months],
        'MONTH': [(month - 1) % 12 + 1 for month in months]
    })

    # Predict sales values and quantities
    sales_predictions = model_sales.predict(prediction_df)
    qty_predictions = model_qty.predict(prediction_df)

    # Ensure non-negative predictions for quantities
    qty_predictions = np.clip(qty_predictions, 0, None)

    # Create forecast dates
    forecast_dates = [pd.Timestamp(year=year + (month - 1) // 12, month=(month - 1) % 12 + 1, day=1) for month in months]

    # Plot the results for sales value
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, sales_predictions, marker='o', linestyle='--', label='Forecasted Sales Value')
    plt.title(f'Forecasted Sales Value - {code}')
    plt.xlabel('Date')
    plt.ylabel('Sales Value')
    plt.legend()
    sales_plot_path = os.path.join(plot_dir, f'sales_plot_{code}.png')
    plt.savefig(sales_plot_path)
    plt.close()

    # Plot the results for sales quantity
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, qty_predictions, marker='o', linestyle='--', label='Forecasted Sales Quantity')
    plt.title(f'Forecasted Sales Quantity - {code}')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.legend()
    qty_plot_path = os.path.join(plot_dir, f'qty_plot_{code}.png')
    plt.savefig(qty_plot_path)
    plt.close()

    return sales_predictions, qty_predictions, forecast_dates, f'plots/sales_plot_{code}.png', f'plots/qty_plot_{code}.png'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_code', methods=['GET', 'POST'])
def select_code():
    if request.method == 'POST':
        code_type = request.form['code_type']
        if code_type == 'customer':
            return redirect(url_for('predict_customer'))
        elif code_type == 'material':
            return redirect(url_for('predict_material'))

    return render_template('select_code.html')

@app.route('/predict_customer', methods=['GET', 'POST'])
def predict_customer():
    if request.method == 'POST':
        customer_code = int(request.form['customer_code'])
        start_year = int(request.form['start_year'])
        months_input = request.form['months']
        months = list(map(int, months_input.split(',')))

        # Predict future sales and quantities for customer
        sales_predictions, qty_predictions, forecast_dates, sales_plot_path, qty_plot_path = predict_sales_and_qty(
            sales_model_customer, qty_model_customer, 'CUSTOMER_CODE', customer_code, start_year, months)

        # Render template with data
        return render_template('result_customer.html',
                               sales_plot=sales_plot_path,
                               qty_plot=qty_plot_path,
                               sales_predictions=sales_predictions,
                               qty_predictions=qty_predictions,
                               forecast_dates=forecast_dates)

    return render_template('predict_customer.html')

@app.route('/predict_material', methods=['GET', 'POST'])
def predict_material():
    if request.method == 'POST':
        material_code = int(request.form['material_code'])
        start_year = int(request.form['start_year'])
        months_input = request.form['months']
        months = list(map(int, months_input.split(',')))

        # Predict future sales and quantities for material
        sales_predictions, qty_predictions, forecast_dates, sales_plot_path, qty_plot_path = predict_sales_and_qty(
            sales_model_material, qty_model_material, 'MATERIAL_CODE', material_code, start_year, months)

        # Render template with data
        return render_template('result_material.html',
                               sales_plot=sales_plot_path,
                               qty_plot=qty_plot_path,
                               sales_predictions=sales_predictions,
                               qty_predictions=qty_predictions,
                               forecast_dates=forecast_dates)

    return render_template('predict_materials.html')

@app.route('/plots/<filename>')
def plot_file(filename):
    return send_from_directory(plot_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
