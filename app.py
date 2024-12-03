import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from flask import Flask, render_template, request, redirect, session, jsonify, abort
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps

# Secure Configuration Management
class Config:
    """Application Configuration Management"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32)
    MONGO_URI = os.environ.get('MONGO_URI')
    #"mongodb+srv://mrvortex911:Vortex%4007@vortex.fr0lbf8.mongodb.net/diabetic_retinopathy?retryWrites=true&w=majority&appName=vortex")
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH','models/Updated-Xception-diabetic-retinopathy.h5' )
    
    # Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application Factory
def create_app(config_class=Config):
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Apply configuration
    app.config.from_object(config_class)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
    
    # Security Middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    # Initialize Extensions
    mongo = PyMongo(app)
    bcrypt = Bcrypt(app)
    CORS(app)
    
    # Rate Limiting
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["100 per day", "30 per hour"]
    )
    
    # Machine Learning Model Loading
    def load_ml_model():
        """Safely load machine learning model"""
        try:
            model = load_model(config_class.MODEL_PATH)
            logger.info(f"Model loaded successfully from {config_class.MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    ml_model = load_ml_model()

    # Prediction Utility Functions
    def predict_dr_stage(image_path: str) -> Dict[str, Any]:
        """Advanced DR Stage Prediction with Confidence"""
        try:
            img = load_img(image_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)
            
            predictions = ml_model.predict(img_array)
            dr_stage_index = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions[0]))
            
            stage_labels = {
                0: 'No DR',
                1: 'Mild DR',
                2: 'Moderate DR', 
                3: 'Severe DR',
                4: 'Proliferative DR'
            }
            
            return {
                'stage': stage_labels[dr_stage_index],
                'confidence': confidence * 100,
                'details': {label: float(prob) * 100 for label, prob in zip(stage_labels.values(), predictions[0])}
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    # Authentication Decorators
    def login_required(f):
        """Authentication decorator for routes"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                logger.warning("Unauthorized access attempt")
                return redirect('/login')
            return f(*args, **kwargs)
        return decorated_function

    # Utility function for checking allowed file extensions
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in config_class.ALLOWED_EXTENSIONS

    # Route Definitions
    @app.route('/')
    def index():
        """Landing Page"""
        return render_template('index.html')

    @app.route('/register', methods=['GET', 'POST'])
    @limiter.limit("10 per minute")
    def register():
        """User Registration with Enhanced Security"""
        if request.method == 'POST':
            user_data = {
                'username': request.form.get('username'),
                'email': request.form.get('email'),
                'password': bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8'),
                'created_at': datetime.utcnow(),
                'last_login': None
            }
            
            # Validate input
            if not all([user_data['username'], user_data['email'], user_data['password']]):
                return jsonify({"error": "Missing required fields"}), 400
            
            # Check existing user
            existing_user = mongo.db.users.find_one({'$or': [
                {'username': user_data['username']},
                {'email': user_data['email']}
            ]})
            
            if existing_user:
                return jsonify({"error": "User already exists"}), 409
            
            # Insert new user
            mongo.db.users.insert_one(user_data)
            logger.info(f"User registered: {user_data['username']}")
            return redirect('/login')
        
        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    @limiter.limit("10 per minute")
    def login():
        """Secure User Login with Logging"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            user = mongo.db.users.find_one({'username': username})
            
            if user and bcrypt.check_password_hash(user['password'], password):
                session['user_id'] = str(user['_id'])
                session.permanent = True
                
                # Update last login
                mongo.db.users.update_one(
                    {'_id': user['_id']},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
                
                logger.info(f"User logged in: {username}")
                return redirect('/predict')
            
            logger.warning(f"Failed login attempt: {username}")
            return jsonify({"error": "Invalid credentials"}), 401
        
        return render_template('login.html')

    @app.route('/predict', methods=['GET', 'POST'])
    @login_required
    def predict():
        """DR Prediction Endpoint with Advanced Error Handling"""
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    prediction = predict_dr_stage(filepath)
                    
                    # Log prediction
                    mongo.db.predictions.insert_one({
                        'user_id': session['user_id'],
                        'prediction': prediction,
                        'timestamp': datetime.utcnow()
                    })
                    
                    return render_template('result.html', prediction=prediction)
                
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    return jsonify({"error": "Prediction processing failed"}), 500
            else:
                return jsonify({"error": "Invalid file type"}), 400
        
        return render_template('predict.html')

    @app.route('/logout')
    def logout():
        """Secure User Logout"""
        session.clear()
        return redirect('/')

    # Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Server Error: {error}")
        return render_template('500.html'), 500

    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app

# Application Entry Point

# Requirements for requirements.txt
"""
flask
flask-pymongo
flask-bcrypt
flask-limiter
flask-cors
tensorflow
numpy
werkzeug
python-dotenv
gunicorn
pillow
"""