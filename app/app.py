from flask import Flask, render_template, request, redirect, url_for, session, flash
import yfinance as yf
import plotly.graph_objs as go
import plotly.offline as pyo
import requests
from textblob import TextBlob
import smtplib
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sqlite3
from flask import jsonify
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'supersecretkey'

# === Setup SQLite database for users ===
DB_NAME = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Drop the existing users table (if it exists)
    c.execute('DROP TABLE IF EXISTS users')

    # Create the new users table with the correct schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
# Load the pre-trained LSTM model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stock_price_model.h5')
model = load_model(MODEL_PATH)


us_stocks = [
    ("MSFT", "Microsoft"), ("AAPL", "Apple"), ("GOOGL", "Alphabet"), ("AMZN", "Amazon"),
    ("META", "Meta"), ("TSLA", "Tesla"), ("NVDA", "NVIDIA"), ("NFLX", "Netflix"),
    ("BRK-B", "Berkshire Hathaway"), ("JNJ", "Johnson & Johnson"), ("JPM", "JPMorgan Chase"),
    ("V", "Visa"), ("MA", "Mastercard"), ("WMT", "Walmart"), ("PG", "Procter & Gamble"),
    ("DIS", "Walt Disney"), ("PFE", "Pfizer"), ("XOM", "Exxon Mobil"), ("CVX", "Chevron"),
    ("BAC", "Bank of America"), ("INTC", "Intel"), ("CSCO", "Cisco"), ("ORCL", "Oracle"),
    ("IBM", "IBM"), ("PEP", "PepsiCo"), ("KO", "Coca-Cola")
]

indian_stocks = [
    ("TCS.NS", "Tata Consultancy Services"), ("INFY.NS", "Infosys"), ("RELIANCE.NS", "Reliance Industries"),
    ("HDFCBANK.NS", "HDFC Bank"), ("ICICIBANK.NS", "ICICI Bank"), ("SBIN.NS", "State Bank of India"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank"), ("HINDUNILVR.NS", "Hindustan Unilever"),
    ("ITC.NS", "ITC Limited"), ("ASIANPAINT.NS", "Asian Paints"), ("MARUTI.NS", "Maruti Suzuki"),
    ("BAJFINANCE.NS", "Bajaj Finance"), ("LT.NS", "Larsen & Toubro"), ("HCLTECH.NS", "HCL Technologies"),
    ("ADANIGREEN.NS", "Adani Green Energy"), ("ADANIPORTS.NS", "Adani Ports"),
    ("ADANIENT.NS", "Adani Enterprises"), ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ("SUNPHARMA.NS", "Sun Pharma"), ("ULTRACEMCO.NS", "UltraTech Cement"),
    ("TATAMOTORS.NS", "Tata Motors"), ("TATASTEEL.NS", "Tata Steel"), ("BHARTIARTL.NS", "Bharti Airtel")
]

european_stocks = [
    ("SAP.DE", "SAP"), ("BAS.DE", "BASF"), ("SIE.DE", "Siemens"), ("VOW3.DE", "Volkswagen"),
    ("AIR.PA", "Airbus"), ("OR.PA", "L'Oréal"), ("MC.PA", "LVMH"), ("SAN.PA", "Sanofi")
]


@app.route('/')
def intro():
    return render_template('intro.html')


# === Registration Route ===
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()  # get the JSON data sent by fetch

    username = data.get('name')  # or 'username' if you prefer
    email = data.get('email')
    password = data.get('password')

    if username and password and email:
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            hashed_password = generate_password_hash(password)
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
            conn.commit()
            conn.close()
            return jsonify({"success": True, "message": "Registration successful! Please log in."}), 200
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "Username already exists. Go to login."}), 400
    else:
        return jsonify({"success": False, "message": "Please provide name, email, and password."}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "message": "Invalid request, no data provided."}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"success": False, "message": "Please provide both email and password."}), 400

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):  # Assuming user[2] is the hashed password
            session['logged_in'] = True
            session['username'] = user[1]  # Assuming user[1] is the username
            return jsonify({"success": True, "message": "Login successful!"}), 200
        else:
            return jsonify({"success": False, "message": "Invalid email or password."}), 401
    except sqlite3.Error as e:
        return jsonify({"success": False, "message": "Database error occurred: " + str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('intro'))

def predict_stock(symbol):
    data = yf.download(symbol, period="10y", interval="1d")
    data = data[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train/test (80% train)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Predict on test set
    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prepare dates for the test data points
    test_dates = data.index[look_back + train_size:]
    test_dates = [str(date.date()) for date in test_dates]

    # Also predict next day price (last 60 days input)
    last_60_days = scaled_data[-look_back:]
    X_last = np.array([last_60_days])
    X_last = np.reshape(X_last, (X_last.shape[0], X_last.shape[1], 1))
    next_day_pred_scaled = model.predict(X_last)
    next_day_pred = scaler.inverse_transform(next_day_pred_scaled)[0][0]

    return next_day_pred, test_dates, actual.flatten().tolist(), predicted.flatten().tolist()


@app.route('/index', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('intro'))

    username = session.get('username', 'User')

    stock_data = None
    plot_div = None
    sentiment = None
    headlines = []
    predicted_price = None
    symbol = None
    sentiment_data = []
    comparison_data = []

    if request.method == 'POST':
        symbol = request.form['symbol']
        session['selected_symbol'] = symbol  # Save selected symbol
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='30d')
        stock_data = hist.reset_index().to_dict(orient='records')

        # Call your modified predict_stock function that returns multiple values
        next_day_pred, test_dates, actual_prices, predicted_prices = predict_stock(symbol)

        # Prepare data for comparison table or graph
        comparison_data = []
        for date, actual, pred in zip(test_dates, actual_prices, predicted_prices):
            comparison_data.append({'date': date, 'actual': actual, 'predicted': pred})

        predicted_price = next_day_pred

        latest_close = hist['Close'][-1]
        if latest_close > 200:
            send_email_alert(
                'user@example.com',
                f'Price Alert for {symbol.upper()}',
                f'{symbol.upper()} has crossed $200! Current price: {latest_close}'
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
        fig.update_layout(
            title=f'{symbol.upper()} Closing Prices (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)'
        )
        plot_div = pyo.plot(fig, output_type='div')

        headlines, sentiment, sentiment_data = get_news_sentiment(symbol)

    else:
        # GET method — only use symbol for dropdown pre-select, not to fetch data
        symbol = session.get('selected_symbol')

    return render_template(
        'index.html',
        username=username,
        symbol=symbol,
        stock_data=stock_data,
        plot_div=plot_div,
        headlines=headlines,
        predicted_price=predicted_price,
        sentiment=sentiment,
        sentiment_data=sentiment_data,
        us_stocks=us_stocks,
        indian_stocks=indian_stocks,
        european_stocks=european_stocks,
        comparison_data=comparison_data
    )


@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    subject = f'New Contact Message from {name}'
    body = f'''
    Name: {name}
    Email: {email}

    Message:
    {message}
    '''

    try:
        send_email_alert('visionstock621@gmail.com', subject, body)
        return jsonify({'success': True, 'message': 'Message sent successfully!', 'category': 'success'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to send message: {str(e)}', 'category': 'danger'})

NEWSAPI_KEY = '87cb5c1a4c4e438ab157532ce518dbf1'

def get_news_sentiment(symbol):
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}&language=en&pageSize=100'
    response = requests.get(url)
    data = response.json()

    headlines = []
    daily_counts = {}

    if data['status'] == 'ok':
        for article in data['articles']:
            title = article['title']
            published_date = article['publishedAt'][:10]
            headlines.append(title)

            blob = TextBlob(title)
            polarity = blob.sentiment.polarity

            if published_date not in daily_counts:
                daily_counts[published_date] = {'positive': 0, 'negative': 0, 'total': 0}

            if polarity > 0:
                daily_counts[published_date]['positive'] += 1
            elif polarity < 0:
                daily_counts[published_date]['negative'] += 1

            daily_counts[published_date]['total'] += 1

        sentiment_data = []
        last_5_dates = sorted(daily_counts.keys())[-5:]
        for date in last_5_dates:
            counts = daily_counts[date]
            total = counts['total']
            sentiment_data.append({
                'date': date,
                'positive': (counts['positive'] / total) * 100 if total else 0,
                'negative': (counts['negative'] / total) * 100 if total else 0,
            })

        total_positive = sum(c['positive'] for c in daily_counts.values())
        total_negative = sum(c['negative'] for c in daily_counts.values())

        overall = 'Positive' if total_positive > total_negative else 'Negative' if total_negative > total_positive else 'Neutral'


        return headlines[:5], overall, sentiment_data

    return [], 'No data', []

def send_email_alert(to_email, subject, body):
    from_email = 'visionstock621@gmail.com'
    password = 'xzvv cahg sikl twpy'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(from_email, password)
    server.sendmail(from_email, [to_email], msg.as_string())
    server.quit()


app = Flask(__name__)
if __name__ == '__main__':
    # init_db()
    app.run(debug=True)
