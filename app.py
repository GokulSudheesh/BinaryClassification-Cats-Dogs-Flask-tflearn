from flask import Flask, render_template, url_for, request, redirect
# url_for looks for the function from the template and the route it corresponds to
from flask_bootstrap import Bootstrap
import os
from classifier import recognize
# https://opensource.com/article/18/4/flask

app = Flask(__name__)
Bootstrap(app)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            class_name = recognize(image_path)
            print("Class: ",class_name)
            result = {
                'class_name': class_name,
                'image_path': image_path
            }
            return render_template('show.html', result = result)
    return render_template('index.html')

app.run(debug = True)
print('yay!')