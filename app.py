from flask import Flask, request, render_template, send_file
from io import BytesIO
import os
from src.features.datadm.cityscape_inference import datadm_inference, visualize_segments

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_prompts = request.form['prompt']
        os.makedirs("temp", exist_ok=True)
        image_path = datadm_inference(input_prompts, "temp")
        result_image = visualize_segments(image_path)
        return send_file(result_image, mimetype='image/png')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
