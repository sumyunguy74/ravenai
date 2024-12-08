from flask import Flask, jsonify, request, send_from_directory, send_file, make_response
from flask_cors import CORS, cross_origin
import torch
from diffusers import StableDiffusionPipeline
import os
import logging
import traceback
import omegaconf

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    negative_prompt = data.get('negativePrompt')

    # Placeholder for generated image path
    image_path = "path/to/generated/image.jpg"

    return jsonify({"imagePath": image_path})

if __name__ == "__main__":
    app.run(debug=True, port=8001)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(RAVEN)

app = Flask(RAVEN, static_folder='../frontend', static_url_path='/static')
CORS(app, resources={r"*": {"origins": "*"}}, supports_credentials=True)

try:
    logger.info("Starting model initialization process...")
    model_directory = os.path.join(os.path.expanduser('~'), 'Documents', 'RAVEN', 'BLACKFORESTIMAGEMODEL')
    model_path = os.path.join(model_directory, 'flux1-dev.safetensors')
    
    logger.info(f"Model directory: {model_directory}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    logger.info(f"Model file size: {os.path.getsize(model_path) / (1024*1024*1024):.2f} GB")
    
    # Log available CUDA devices
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    logger.info("Attempting to initialize pipeline...")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        load_safety_checker=False,
        torch_dtype=torch.float32,  # Explicitly set dtype
        variant="fp32"  # Use fp32 variant
    )
    
    logger.info("Pipeline created, moving to CPU...")
    pipe = pipe.to("cpu")
    logger.info("Pipeline initialized successfully and moved to CPU.")
    
except Exception as e:
    logger.error("Error during model initialization:")
    logger.error(traceback.format_exc())
    logger.error(f"Exception type: {type(e)}")
    logger.error(f"Exception args: {e.args}")
    pipe = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_images():
    if pipe is None:
        return jsonify({"error": "Model initialization failed"}), 503

    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        # Generate image
        image = pipe(prompt).images[0]
        
        # Save the generated image
        os.makedirs(os.path.join(app.static_folder, 'generated_images'), exist_ok=True)
        image_path = os.path.join(app.static_folder, 'generated_images', 'generated.png')
        image.save(image_path)
        
        return jsonify({"image": "/static/generated_images/generated.png"})
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-flux', methods=['POST'])
def generate_flux_image():
    global pipe
    
    # Ensure model is initialized
    if pipe is None:
        logger.error("Model not initialized")
        return jsonify({"error": "Model initialization failed"}), 503

    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No input data provided"}), 400

        prompt = data.get('prompt', "A cat holding a sign that says hello world")
        logger.info(f"Generating flux image with prompt: {prompt}")

        # Use more memory-efficient parameters
        try:
            with torch.no_grad():  # Reduce memory usage
                image = pipe(
                    prompt,
                    height=512,  # Further reduced from 768
                    width=512,   # Further reduced from 768
                    guidance_scale=3.5,
                    num_inference_steps=20,  # Further reduced from 30
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory error")
                return jsonify({"error": "GPU memory exceeded. Try a shorter prompt or contact administrator."}), 507
            raise e

        # Ensure the generated_images directory exists
        save_dir = os.path.join(app.static_folder, 'generated_images')
        os.makedirs(save_dir, exist_ok=True)

        # Save the generated image
        image_path = os.path.join(save_dir, "flux-dev.png")
        logger.info(f"Saving image to: {image_path}")
        image.save(image_path)

        return jsonify({"image": "/static/generated_images/flux-dev.png"})

    except Exception as e:
        logger.error(f"Error in generate_flux_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to generate image: {str(e)}"}), 500

@app.route('/generate-fill', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_fill_image():
    if pipe is None:
        response = jsonify({"error": "Model initialization failed"}), 503
        return response

    try:
        data = request.json
        prompt = data.get('prompt', "A cat holding a sign that says hello world")
        
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        
        # Save the generated image
        image_path = os.path.join(app.static_folder, 'generated_images', "fill-dev.png")
        logger.info(f"Saving image to: {image_path}")
        image.save(image_path)
        
        response = jsonify({"image": "/static/generated_images/fill-dev.png"})
        return response
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        response = jsonify({"error": str(e)}), 500
        return response

@app.after_request
def after_request(response):
    logger.info("Adding CORS headers to response")  # Log CORS header addition
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    logger.info("CORS headers added to response")  # Log CORS header addition
    return response

@app.before_request
def log_request_info():
    logger.info(f"Request Headers: {request.headers}")
    logger.info(f"Request Body: {request.get_data()}")

@app.after_request
def log_response_info(response):
    try:
        # Only log response data for JSON responses
        if response.mimetype == 'application/json':
            logger.info(f"Response Status: {response.status}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            try:
                # Try to get JSON data
                data = response.get_json()
                logger.info(f"Response Body: {data}")
            except Exception:
                # If not JSON, just log the content type
                logger.info(f"Response Content-Type: {response.content_type}")
        else:
            logger.info(f"Response Status: {response.status}")
            logger.info(f"Response Content-Type: {response.content_type}")
    except Exception as e:
        logger.error(f"Error logging response: {str(e)}")
    return response

if __name__ == '__main__':
    # Ensure the static folder exists
    os.makedirs(os.path.join(app.static_folder), exist_ok=True)
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=True)
    os.makedirs('uploads', exist_ok=True)
    