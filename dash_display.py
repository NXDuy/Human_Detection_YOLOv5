from flask import Flask, Response, render_template
from detection import HumanDetection

app = Flask(__name__)

webcam_data = HumanDetection(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(webcam_data(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
