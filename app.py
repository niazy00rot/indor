from model import load_model, predict_image, model_info, CLASSES, device
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        'status': 'success',
        'info': model_info,
        'classes': CLASSES
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No image selected'}), 400
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        predictions = predict_image(image)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        print("Loading model...")
        load_model()
        print("Model loaded successfully!")
        print(f"Device: {device}")
        print(f"Ready to classify {len(CLASSES)} indoor scene types")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'best_resnet50_model.pth' is in the same directory")