import os
import io
import base64
import json
import tempfile
import time
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from translation_pipeline import MangaTranslationPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the manga translation pipeline
translation_pipeline = MangaTranslationPipeline(
    azure_key=os.getenv('AZURE_VISION_KEY'),
    azure_endpoint=os.getenv('AZURE_VISION_ENDPOINT'),
    yolo_model_path=os.getenv('YOLO_MODEL_PATH'),
    font_path=os.getenv('FONT_PATH'),
    deepl_api_key=os.getenv('DEEPL_API_KEY'),
    debug=True
)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })

@app.route('/translate', methods=['POST'])
def translate_image():
    """
    Translate manga/manhwa image endpoint.
    
    Expects a JSON payload with:
    - image: base64-encoded image data
    - targetLang: target language code (e.g., 'en', 'es')
    - imageInfo: optional metadata about the image
    
    Returns:
    - translatedImage: base64-encoded translated image
    - info: processing information
    """
    try:
        # Get request data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'Missing required field: image'
            }), 400
        
        # Get target language (default to English)
        target_lang = data.get('targetLang', 'en-gb')
        
        # Extract base64 image data
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix (e.g., "data:image/png;base64,")
            image_data = image_data.split(',', 1)[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        # Set the target language for translation
        translation_pipeline.text_replacer.target_language = target_lang
        
        # Create temporary output path
        output_path = temp_path + '.translated.png'
        
        # Process the image
        result_image, info = translation_pipeline.process(temp_path, output_path)
        
        # Convert result to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Clean up temporary files
        try:
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {e}")
        
        # Return response
        return jsonify({
            'translatedImage': f'data:image/png;base64,{encoded_image}',
            'info': {
                'processingTime': info.get('total_time', 0),
                'textBoxesCount': info.get('steps', {}).get('ocr', {}).get('detected_boxes', 0),
                'translatedBoxesCount': info.get('steps', {}).get('replacement', {}).get('translated_boxes', 0)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    """
    Get supported languages.
    
    Returns a list of available target languages.
    """
    return jsonify({
        'languages': [
            {'code': 'en', 'name': 'English'},
            {'code': 'es', 'name': 'Spanish'},
            {'code': 'fr', 'name': 'French'},
            {'code': 'de', 'name': 'German'},
            {'code': 'it', 'name': 'Italian'},
            {'code': 'pt', 'name': 'Portuguese'},
            {'code': 'ru', 'name': 'Russian'},
            {'code': 'zh', 'name': 'Chinese'},
            {'code': 'ja', 'name': 'Japanese'},
            {'code': 'ko', 'name': 'Korean'}
        ]
    })

if __name__ == '__main__':
    # Get port from environment or use default 5000
    port = int(os.getenv('PORT', 5000))
    
    # Start the API server
    app.run(host='0.0.0.0', port=port, debug=True) 