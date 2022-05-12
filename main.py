from flask import Flask, render_template, Response
from utils.Camera import videoGetter
import os
app = Flask(__name__)

cameras = [
    'Datasets\VIRAT_S_000002.mp4',
    'Datasets\VIRAT_S_000200_00_000100_000171.mp4',
    'Datasets\VIRAT_S_000200_01_000226_000268.mp4'
]

@app.route('/video_feed/<route>', methods=['GET'])
def video_feed(route):
    route = route.replace(os.sep, '/')
    return Response(videoGetter(route), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', len=len(cameras), cameras=cameras)

if __name__ == '__main__':
    app.run()
