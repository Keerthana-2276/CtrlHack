from flask import Flask, render_template, redirect, url_for
import subprocess
import os

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# Ensure necessary folders exist
required_dirs = ['static', 'static/css1', 'static/js1', 'static/images1']
for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# Home Route
@app.route('/')
def home():
    contact_data = {}  # Provide a default empty dictionary
    return render_template('index.html', contact_data=contact_data)

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/contact')
def contact():
    contact_data = {}  # Provide an empty dictionary here as well
    return render_template('contact.html', contact_data=contact_data)

# Start Exercise Route (Runs camera4.py)
@app.route('/start-exercise')
def start_exercise():
    try:
        subprocess.Popen(["python", "camera4.py"])
    except Exception as e:
        print(f"Error launching camera4.py: {e}")
    return redirect(url_for('home'))

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
