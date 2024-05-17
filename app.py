import numpy as np
import pandas as pd
import os
from flask import Flask, jsonify, render_template, request, redirect, flash, send_file, url_for, session, make_response, current_app
from datetime import timedelta
from flask_mysqldb import MySQL
import mysql.connector
from werkzeug.utils import secure_filename
import pickle
import urllib.request
from datetime import datetime
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from email.mime.text import MIMEText
import time
import threading
import smtplib




basedir = os.path.abspath(os.path.dirname(__file__))

intrusion = pickle.load(open('models/H5_PKL/intrusion.pkl', 'rb'))
mlp_unsw = pickle.load(open('unsw/model/mlp_unsw_flask.pkl', 'rb'))
kddDnnModel = load_model('models/H5_PKL/kddDnnFinal.h5')

with open('models/H5_PKL/outcomesFinal.pkl', 'rb') as f:
    outcomes = pickle.load(f)

progress = 0

conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
cursor = conn.cursor()

app = Flask(__name__)
app.secret_key = "secret key"
app.permanent_session_lifetime = timedelta(days=1)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB limit

app.config.update(
    UPLOADED_PATH = os.path.join(basedir, 'uploads'),
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    ALLOWED_EXTENSIONS = set(['.csv, .xlsx'])
    )

MAX_FILE_SIZE = 25 * 1024 * 1024 



@app.route("/")
def index():
    if 'username' in session:
        return render_template('index.html', username = session['username'])
    else:
        return render_template('index.html')
    
    

@app.route("/login", methods=['GET', 'POST'])
def login():
    msg=''
    if request.method=='POST':
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
        cursor = conn.cursor()
        session.permanent = True
        username = request.form['username']
        password = request.form['password']
        cursor.execute('SELECT * FROM tbl_users WHERE username=%s AND password=%s', (username, password))
        record = cursor.fetchone()
        conn.close()
        cursor.close()
        if record:
            session['loggedin'] = True
            session['username'] = record[1]
            return redirect(url_for('index')) 
        else:
            msg = 'Incorrect username or password. Try Again!'
    return render_template('login.html', msg=msg)

#########################################################################################################################

@app.route("/register", methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="ids")
        cursor = conn.cursor()
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            msg = 'Passwords do not match!'
        else:
            cursor.execute('SELECT * FROM tbl_users WHERE username=%s', (username,))
            record = cursor.fetchone()
            if record:
                msg = 'Username already exists!'
            else:
                cursor.execute('INSERT INTO tbl_users (username, password) VALUES (%s, %s)', (username, password))
                conn.commit()
                msg = 'You have successfully registered!'
        conn.close()
        cursor.close()
        if msg == 'You have successfully registered!':
            return redirect(url_for('login'))  # Redirect to login page after successful registration
    return render_template('register.html', msg=msg)



#########################################################################################################################


@app.route('/about', methods=['GET', 'POST'])
def about():
        if 'username' in session:
         return render_template('about.html', username = session['username'])
        else:
            return render_template('about.html')
 # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
@app.route('/contact', methods=['GET', 'POST'])
def contact():
        if 'username' in session:
         return render_template('contact.html', username = session['username'])
        else:
            return render_template('contact.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        message = request.form['message']

        # Send email
        send_email(first_name, last_name, email, message)
        return 'Form submitted successfully!'

def send_email(first_name, last_name, email, message):
    sender_email = "yusuf.msalem@gmail.com"  # Change this to your email address
    receiver_email = "yusuf.msalem@gmial.com"  # Change this to recipient email address
    subject = "Contact Form Submission"
    body = f"First Name: {first_name}\nLast Name: {last_name}\nEmail: {email}\nMessage: {message}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, "jlog qusb dhlb ckfu")  # Change this to your email password
    smtp_server.sendmail(sender_email, receiver_email, msg.as_string())
    smtp_server.quit()


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/graphDT', methods=['GET', 'POST'])
def graphDT():
    if 'username' in session:
        return render_template('graphDT.html', username = session['username'])
    else:
        return render_template('graphDT.html')
    
    
@app.route('/graphMLP', methods=['GET', 'POST'])
def graphMLP():
    if 'username' in session:
        return render_template('graphMLP.html', username = session['username'])
    else:
        return render_template('graphMLP.html')
    
    
@app.route('/graphDNN', methods=['GET', 'POST'])
def graphDNN():
    if 'username' in session:
        return render_template('graphDNN.html', username = session['username'])
    else:
        return render_template('graphDNN.html')
        
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/select_dataset")
def select_dataset():
    if 'username' in session:
        return render_template('select_dataset.html', username = session['username'])
    else:
        return render_template('login.html')
    

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


@app.route('/uploadCsv', methods=['GET', 'POST'])
def uploadCsv():
    global progress
    if request.method == 'POST':
        progress = 0  # Reset progress at the start of each upload
        file = request.files['dataset']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        file_size = os.path.getsize(file_path)

        if not filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file.'}), 400
        
        elif file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File size exceeds the limit (25MB).'}), 400

        threading.Thread(target=train_model, args=(file_path,)).start()
        
        return jsonify({'message': 'File uploaded and model training started'}), 202
    
    if 'loggedin' in session:
        return render_template('uploadCsv.html', username=session['username'])
    else:
        return render_template('login.html')

classification_report_str = ""

def train_model(file_path):
    with app.app_context():
        global progress, classification_report_str
        data = pd.read_csv(file_path)
        
        # Automatically detect and label encode categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])
        
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
        
        # Determine the total number of iterations (for simplicity, let's assume 500)
        total_iterations = mlp.max_iter
            # Update the model with a partial fit
        for i in range(total_iterations):
            mlp.partial_fit(X_train, y_train, classes=np.unique(y))
            
            progress = int((i + 1) / total_iterations * 100)

        # for i in range(total_iterations):
        #     mlp.fit(X_train, y_train)
            
        #     progress = int((i + 1) / total_iterations * 100)

        pickle.dump(mlp, open('mlp_model.pkl', 'wb'))

        predictions = mlp.predict(X_test)

        report = classification_report(y_test, predictions)
        
        classification_report_str = report

        progress = 100


@app.route('/show_classification_report')
def show_classification_report():
    if 'loggedin' in session:
        if classification_report_str:
            return render_template('classification_report.html', report=classification_report_str, username=session['username'])
        else:
            return render_template('classification_report.html', report='Classification report not available.', username=session['username'])
    else:
        return render_template('login.html')

@app.route('/progress')
def get_progress():
    return jsonify({'progress': progress})


################################################################################################################


@app.route('/kddPredictionDNN', methods = ['GET', 'POST'])
def kddPredictionDNN():
    if 'loggedin' in session:
        return render_template('kddPredictionDNN.html', username = session['username'])
    else:
        return render_template('login.html')
    
    

@app.route('/kddPredictionDT', methods = ['GET', 'POST'])
def kddPredictionDT():
    if 'loggedin' in session:
        return render_template('kddPredictionDT.html', username = session['username'])
    else:
        return render_template('login.html')
    

@app.route('/unswPredictionMLP', methods = ['GET', 'POST'])
def unswPredictionMLP():
    if 'loggedin' in session:
        return render_template('unswPredictionMLP.html', username = session['username'])
    else:
        return render_template('login.html')    


@app.route('/DNNpredict', methods=['POST'])
def DNNpredict():
    int_feature = [float(x) for x in request.form.values()]
  
    final_features = np.array([int_feature])
   
    result = kddDnnModel.predict(final_features)
    predicted_label = outcomes[np.argmax(result)]  # Convert prediction to original label

    if 'loggedin' in session:
        return render_template('kddPredictionDNN.html', prediction_text=predicted_label, username=session['username'])
    else:
        return render_template('login.html')
    
    
@app.route('/DTpredict', methods=['POST'])
def DTpredict():
    int_feature = [x for x in request.form.values()]
    
    final_features = [np.array(int_feature)]
    
    result = intrusion.predict(final_features)
    for i in result:
        print(i, end="")
     
    if 'loggedin' in session:
        return render_template('kddPredictionDT.html', prediction_text=i, username=session['username'])
    else:
        return render_template('login.html')
    

@app.route('/MLPpredict', methods=['POST'])
def MLPpredict():
    int_feature = [float(x) for x in request.form.values()]
    
    final_features = [np.array(int_feature)]
    
    result = mlp_unsw.predict(final_features)
    for i in result:
        print(i, end="")
     
    if 'loggedin' in session:
        return render_template('unswPredictionMLP.html', prediction_text=i, username=session['username'])
    else:
        return render_template('login.html')

    


@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(debug=True)