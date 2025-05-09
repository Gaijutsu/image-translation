// Global variables
let isEnabled = false;
let originalImages = new Map(); // Store original images data
let translatingImages = new Set(); // Track images being translated
let observer = null; // For watching DOM changes

// Initialize when content script loads
initialize();

function initialize() {
  // Check if translation is enabled
  browser.storage.local.get('enabled').then(result => {
    isEnabled = result.enabled || false;
    
    if (isEnabled) {
      startTranslation();
    }
  });
  
  // Setup message listener
  browser.runtime.onMessage.addListener(handleMessages);
}

// Handle messages from popup or background script
function handleMessages(message) {
  switch (message.action) {
    case 'enableTranslation':
      isEnabled = true;
      startTranslation();
      break;
      
    case 'disableTranslation':
      isEnabled = false;
      stopTranslation();
      break;
      
    case 'translationResult':
      handleTranslationResult(message);
      break;
  }
}

// Start looking for manga images and translating them
function startTranslation() {
  // Process existing images first
  processExistingImages();
  
  // Setup observer for new images
  setupObserver();
}

// Stop translation and restore original images
function stopTranslation() {
  // Disconnect observer
  if (observer) {
    observer.disconnect();
    observer = null;
  }
  
  // Restore all original images
  restoreOriginalImages();
}

// Process all existing images on the page
function processExistingImages() {
  const images = document.querySelectorAll('img');
  images.forEach(image => {
    processImage(image);
  });
}

// Setup mutation observer to watch for new images
function setupObserver() {
  if (observer) {
    observer.disconnect();
  }
  
  observer = new MutationObserver(mutations => {
    for (const mutation of mutations) {
      // Look for added nodes
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        mutation.addedNodes.forEach(node => {
          // Check if node is an image
          if (node.nodeName === 'IMG') {
            processImage(node);
          }
          
          // Check children for images
          if (node.getElementsByTagName) {
            const images = node.getElementsByTagName('img');
            Array.from(images).forEach(img => {
              processImage(img);
            });
          }
        });
      }
    }
  });
  
  // Start observing the document
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Process a single image
function processImage(imgElement) {
  // Skip if already processed or currently translating
  if (imgElement.dataset.translating === 'true' || imgElement.dataset.translated === 'true') {
    return;
  }
  
  // Check if image might be manga/manhwa
  if (isMangaImage(imgElement)) {
    // Mark image as being processed
    imgElement.dataset.translating = 'true';
    
    // Add to tracking set
    const imgId = Date.now() + '-' + Math.random().toString(36).substring(2, 9);
    imgElement.dataset.translationId = imgId;
    translatingImages.add(imgId);
    
    // Save original image info
    saveOriginalImage(imgElement);
    
    // Send image for translation
    sendImageForTranslation(imgElement);
    
    // Optional: Show loading indicator
    showLoadingIndicator(imgElement);
  }
}

// Check if an image is likely to be manga/manhwa
function isMangaImage(imgElement) {
  // Skip small images
  if (imgElement.width < 200 || imgElement.height < 200) {
    return false;
  }
  
  // Check aspect ratio (manga panels are often portrait or square)
  const ratio = imgElement.width / imgElement.height;
  if (ratio > 2.5) { // Very wide images are unlikely to be manga panels
    return false;
  }
  
  // Check URL patterns common for manga sites
  const src = imgElement.src.toLowerCase();
  if (src.includes('banner') || src.includes('logo') || src.includes('icon') || src.includes('avatar')) {
    return false;
  }
  
  // Add more specific checks based on known manga sites if needed
  
  return true;
}

// Save original image information
function saveOriginalImage(imgElement) {
  const id = imgElement.dataset.translationId;
  originalImages.set(id, {
    src: imgElement.src,
    srcset: imgElement.srcset,
    style: imgElement.getAttribute('style'),
    width: imgElement.width,
    height: imgElement.height
  });
}

// Send image to background script for translation
async function sendImageForTranslation(imgElement) {
  try {
    // Get image data
    const imageData = await getBase64FromImageElement(imgElement);
    
    // Build image info object
    const imageInfo = {
      id: imgElement.dataset.translationId,
      width: imgElement.width,
      height: imgElement.height,
      url: imgElement.src
    };
    
    // Send to background script
    browser.runtime.sendMessage({
      action: 'translateImage',
      imageData: imageData,
      imageInfo: imageInfo
    });
  } catch (error) {
    console.error('Error sending image for translation:', error);
    removeLoadingIndicator(imgElement);
    imgElement.dataset.translating = 'false';
  }
}

// Handle translation result from background script
function handleTranslationResult(message) {
  if (!message.originalImageInfo || !message.originalImageInfo.id) {
    return;
  }
  
  // Find the image element
  const imgId = message.originalImageInfo.id;
  const imgElement = document.querySelector(`img[data-translation-id="${imgId}"]`);
  
  if (!imgElement) {
    return;
  }
  
  // Remove from tracking set
  translatingImages.delete(imgId);
  
  // Handle successful translation
  if (message.success && message.translatedImageData) {
    // Replace with translated image
    imgElement.src = message.translatedImageData;
    imgElement.dataset.translated = 'true';
  }
  
  // Remove loading indicator
  removeLoadingIndicator(imgElement);
  imgElement.dataset.translating = 'false';
}

// Restore all original images
function restoreOriginalImages() {
  document.querySelectorAll('img[data-translated="true"]').forEach(img => {
    const id = img.dataset.translationId;
    const original = originalImages.get(id);
    
    if (original) {
      img.src = original.src;
      if (original.srcset) {
        img.srcset = original.srcset;
      }
      if (original.style) {
        img.setAttribute('style', original.style);
      }
    }
    
    // Clear translation flags
    img.dataset.translated = 'false';
    img.dataset.translating = 'false';
    
    // Remove loading indicator if present
    removeLoadingIndicator(img);
  });
}

// Convert image element to base64
function getBase64FromImageElement(imgElement) {
  return new Promise((resolve, reject) => {
    try {
      // For CORS-protected images, we need to draw on canvas
      const canvas = document.createElement('canvas');
      
      // Use natural dimensions to ensure full resolution
      canvas.width = imgElement.naturalWidth;
      canvas.height = imgElement.naturalHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgElement, 0, 0);
      
      // Get as base64, PNG format to maintain quality
      canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      }, 'image/png');
      
    } catch (error) {
      reject(error);
    }
  });
}

// Add loading indicator to image
function showLoadingIndicator(imgElement) {
  // Create container
  const container = document.createElement('div');
  container.classList.add('manga-translator-loading');
  container.style.position = 'absolute';
  container.style.top = '0';
  container.style.left = '0';
  container.style.width = '100%';
  container.style.height = '100%';
  container.style.display = 'flex';
  container.style.alignItems = 'center';
  container.style.justifyContent = 'center';
  container.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  container.style.color = 'white';
  container.style.zIndex = '1000';
  
  // Create spinner
  const spinner = document.createElement('div');
  spinner.textContent = 'Translating...';
  spinner.style.padding = '10px';
  spinner.style.borderRadius = '5px';
  spinner.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  
  container.appendChild(spinner);
  
  // Make parent position relative if not already
  const parent = imgElement.parentElement;
  const originalPosition = window.getComputedStyle(parent).position;
  if (originalPosition === 'static') {
    parent.style.position = 'relative';
    parent.dataset.originalPosition = 'static';
  }
  
  // Add loading indicator after image
  parent.appendChild(container);
  parent.dataset.hasLoadingIndicator = 'true';
}

// Remove loading indicator
function removeLoadingIndicator(imgElement) {
  const parent = imgElement.parentElement;
  
  if (parent && parent.dataset.hasLoadingIndicator === 'true') {
    // Find and remove the loading container
    const loadingContainer = parent.querySelector('.manga-translator-loading');
    if (loadingContainer) {
      parent.removeChild(loadingContainer);
    }
    
    // Reset position if we changed it
    if (parent.dataset.originalPosition === 'static') {
      parent.style.position = 'static';
      delete parent.dataset.originalPosition;
    }
    
    delete parent.dataset.hasLoadingIndicator;
  }
} 