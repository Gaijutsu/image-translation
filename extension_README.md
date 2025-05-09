# Manga Translation Firefox Extension

A Firefox extension that automatically translates manga and manhwa images on webpages by sending them to your translation API server.

## Features

- Automatically detects manga and manhwa images on webpages
- Sends images to your custom translation API
- Replaces original images with translated versions
- Supports multiple target languages
- Toggle translation on/off with a single click
- Minimal UI with easy-to-use settings

## Setup

### Setting up the API Server

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables by creating a `.env` file with:
   ```
   AZURE_VISION_KEY=your_azure_vision_key
   AZURE_VISION_ENDPOINT=your_azure_vision_endpoint
   YOLO_MODEL_PATH=path/to/yolo_model.pt
   FONT_PATH=fonts/your_manga_font.ttf
   DEEPL_API_KEY=your_deepl_api_key
   PORT=5000
   ```

3. Start the API server:
   ```bash
   python api_server.py
   ```

4. The server will be available at `http://localhost:5000`

### Installing the Extension in Firefox

#### Developer Mode (Temporary Installation)

1. Open Firefox and navigate to `about:debugging`
2. Click "This Firefox" in the sidebar
3. Click "Load Temporary Add-on..."
4. Navigate to the extension directory and select the `manifest.json` file
5. The extension is now installed temporarily (until Firefox is closed)

#### Packaging for Distribution

1. Zip the extension files:
   ```bash
   zip -r manga-translator.zip manifest.json icons/ popup/ content_scripts/ background_scripts/
   ```

2. You can then install the .zip file from the Add-ons Manager in Firefox

## Usage

1. Click the extension icon in the toolbar to open the settings popup
2. Toggle the translation on/off with the switch
3. Set your preferred target language
4. Enter the API endpoint (e.g., `http://localhost:5000/translate`)
5. Click "Save Settings"
6. Browse manga/manhwa websites and the images will be automatically translated

## Configuration

The extension has these configurable settings:

- **Translation Toggle**: Enable/disable the translation feature
- **Target Language**: Choose which language to translate to
- **API Endpoint**: URL of your translation API server

## How It Works

1. The extension scans webpages for images that might be manga/manhwa
2. When it finds a suitable image, it sends the image data to your API server
3. The server processes the image using your manga translation pipeline
4. The translated image is sent back to the extension
5. The extension replaces the original image with the translated version

## Advanced Usage

### Custom API Server

You can modify the `api_server.py` file to customize the translation logic:

- Change the translation pipeline parameters
- Add support for more languages
- Add caching for faster response times
- Deploy to a cloud server for public access

### Extension Customization

You can customize the extension by:

- Adding more detection rules in `isMangaImage()` function
- Changing the UI in the popup files
- Adding more settings or features

## Troubleshooting

- **Images Not Being Translated**: Check if the extension is enabled and the API server is running
- **CORS Errors**: Ensure your API server has CORS enabled properly
- **API Connection Issues**: Verify the endpoint URL and check server logs
- **Translation Quality**: Adjust the translation pipeline parameters for better results 