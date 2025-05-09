// Initialize default settings
browser.runtime.onInstalled.addListener(() => {
  browser.storage.local.set({
    enabled: false,
    targetLang: 'EN-GB',
    apiEndpoint: ''
  });
});

// Listen for messages from popup or content scripts
browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle different actions
  switch (message.action) {
    case 'enableTranslation':
      // When translation is enabled
      browser.browserAction.setIcon({
        path: {
          48: '/icons/icon-active-48.png',
          96: '/icons/icon-active-96.png'
        }
      });
      break;
      
    case 'disableTranslation':
      // When translation is disabled
      browser.browserAction.setIcon({
        path: {
          48: '/icons/icon-48.png',
          96: '/icons/icon-96.png'
        }
      });
      break;
      
    case 'settingsUpdated':
      // When settings are updated
      console.log('Settings updated:', message.settings);
      break;
      
    case 'translateImage':
      // When content script requests translation
      translateImage(message.imageData, message.imageInfo, sender.tab.id);
      break;
  }
  
  return true; // Required for async response
});

// Listen for tab updates to inject content script if needed
browser.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only act when page is fully loaded and URL has changed
  if (changeInfo.status === 'complete') {
    // Check if translation is enabled
    browser.storage.local.get('enabled').then(result => {
      if (result.enabled) {
        // Notify content script to start translation
        browser.tabs.sendMessage(tabId, {
          action: 'enableTranslation'
        }).catch(() => {
          // Content script might not be loaded yet, which is fine
        });
      }
    });
  }
});

// Function to handle image translation via API
async function translateImage(imageData, imageInfo, tabId) {
  try {
    // Get settings from storage
    const settings = await browser.storage.local.get(['apiEndpoint', 'targetLang']);
    
    if (!settings.apiEndpoint) {
      throw new Error('API endpoint not set');
    }
    
    // Create request to translation API
    const response = await fetch(settings.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        image: imageData,
        targetLang: settings.targetLang,
        imageInfo: imageInfo
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    // Get translated image data from response
    const result = await response.json();
    
    // Send translated image back to content script
    browser.tabs.sendMessage(tabId, {
      action: 'translationResult',
      success: true,
      originalImageInfo: imageInfo,
      translatedImageData: result.translatedImage
    });
    
  } catch (error) {
    console.error('Translation error:', error);
    
    // Send error to content script
    browser.tabs.sendMessage(tabId, {
      action: 'translationResult',
      success: false,
      originalImageInfo: imageInfo,
      error: error.message
    });
  }
} 