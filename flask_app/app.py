from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import sqlite3
import shutil
import time
from moviepy.editor import VideoFileClip

app = Flask(__name__)

# Global model variable
model = None



# Directories
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
MODELS_FOLDER = "static/models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Database setup
DATABASE = "models.db"


def init_db():
    """Initialize the database and create the models table."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()
    conn.close()

def add_model_to_db(name, path):
    """Add a model to the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO models (name, path) VALUES (?, ?)", (name, path))
    conn.commit()
    conn.close()

def get_models_from_db():
    """Retrieve all models from the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM models")
    models = cursor.fetchall()
    conn.close()
    return models

def get_model_path(model_id):
    """Retrieve the file path of a model by ID."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM models WHERE id = ?", (model_id,))
    path = cursor.fetchone()
    conn.close()
    return path[0] if path else None

# Helper to update the selected YOLO model
def set_selected_model(model_path):
    global model
    try:
        model = YOLO(model_path)
        print(f"Model updated to: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Function to check if model already exists in the database
def model_exists(model_path):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM models WHERE path = ?", (model_path,))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists

init_db()  # Initialize the database

# Load default model
if model is None:
    current_model_path = "D:/ug_mini_proj/flask_app/static/best.pt"
    model = YOLO(current_model_path)

# Routes

# Landing page (homepage)
@app.route('/')
def homepage():
    return render_template('homepage.html')  # Updated to use the new homepage.html

# Detection page
@app.route('/detect', methods=['GET', 'POST'])
def detect_page():
    models = get_models_from_db()  # Pass available models to the detection page
    
    return render_template('detection.html', models=models)

@app.route('/set_model', methods=['POST'])
def set_model():
    selected_model_path = request.form.get('model_path')
    if selected_model_path:
        set_selected_model(selected_model_path)
    return redirect(url_for('detect_page'))

@app.route('/detect_image', methods=['POST'])
def detect_image():
    print(model)
    image_file = request.files['image_file']
    if not image_file or not allowed_file(image_file.filename):
        return "No valid image file uploaded", 400

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    results = model(source=image_path, show=False, save=True, name="result")
    saved_image_path = os.path.join(results[0].save_dir, image_file.filename)
    if os.path.exists(saved_image_path):
        shutil.move(saved_image_path, os.path.join(RESULTS_FOLDER, "result.jpg"))
        return redirect(url_for('show_result', image_name="result.jpg"))
    return "Error: Result image not found", 404

@app.route('/detect_video', methods=['POST'])
def detect_video():
    video_file = request.files['video_file']
    if not video_file or not allowed_file(video_file.filename):
        return "No valid video file uploaded", 400

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    results = model(source=video_path, show=False, save=True)
    result_video_dir = results[0].save_dir
    original_name = os.path.splitext(video_file.filename)[0]
    result_video_filename = f"{original_name}.avi"
    result_video_path = os.path.join(result_video_dir, result_video_filename)

    if os.path.exists(result_video_path):
        final_video_path = os.path.join(RESULTS_FOLDER, result_video_filename)
        shutil.move(result_video_path, final_video_path)
        clip = VideoFileClip(final_video_path)
        clip.write_videofile(os.path.join(RESULTS_FOLDER, f"{original_name}.mp4"), codec="libx264")
        os.remove(final_video_path)
        return redirect(url_for('show_result', result_name=f"{original_name}.mp4", is_video=True))
    return "Error: Video result not found", 404

@app.route('/result')
def show_result():
    result_name = request.args.get('result_name', 'result.jpg')
    is_video = request.args.get('is_video', 'false') == 'True'
    result_url = url_for('static', filename=f'results/{result_name}')
    return render_template('result.html', image_url=result_url, is_video=is_video)

# Training page
@app.route('/training', methods=['GET'])
def training_page():
    return render_template('training.html')  # Placeholder for the training page

@app.route('/train_model', methods=['POST'])
def train_model():
    # Handle uploaded model file and dataset YAML file
    model_file = request.files['model_file']
    data_yaml_file = request.files['data_yaml']
    model_name = request.form['model_name']
    epochs = int(request.form['epochs'])
    workers = int(request.form['workers'])
    
    if model_file and allowed_file(model_file.filename) and data_yaml_file and allowed_file(data_yaml_file.filename):
        # Save the uploaded files
        model_file_path = os.path.join(MODELS_FOLDER, model_file.filename)
        data_yaml_path = os.path.join(UPLOAD_FOLDER, data_yaml_file.filename)
        
        model_file.save(model_file_path)
        data_yaml_file.save(data_yaml_path)

        # Start the training with the uploaded files
        model = YOLO(model_file_path)  # Load the pretrained model
        model.train(data=data_yaml_path, epochs=epochs, batch=16, imgsz=640, freeze=11, workers=workers)

        # Find the directory where the best model is saved (latest training run)
        runs_dir = "runs/detect"
        latest_run_dir = max([os.path.join(runs_dir, d) for d in os.listdir(runs_dir)], key=os.path.getmtime)
        best_model_path = os.path.join(latest_run_dir, 'weights', 'best.pt')

        # Generate a unique model name using a timestamp to avoid name conflicts
        timestamp = int(time.time())
        new_model_name = f"{model_name}_{timestamp}.pt"
        new_model_path = os.path.join(MODELS_FOLDER, new_model_name)
        
        # Check if the model path already exists in the database
        if not model_exists(new_model_path):
            shutil.copy(best_model_path, new_model_path)
            add_model_to_db(new_model_name, new_model_path)
        else:
            return "Model already exists in the database.", 400

        # Get the path to the resulting image (e.g., from the latest detection run)
        result_image_path = os.path.join(latest_run_dir, 'results.png')
        
        # Move the result image to the 'static/results' folder for easy access
        if os.path.exists(result_image_path):
            shutil.move(result_image_path, os.path.join(RESULTS_FOLDER, 'performance.jpg'))

        # Pass the result image and model name to the frontend for display
        return render_template('training.html', result_image='performance.jpg', model_name=new_model_name)
    
    return "Invalid file types. Please upload a valid model and dataset configuration.", 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov','pt','yaml'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
