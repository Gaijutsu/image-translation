document.addEventListener('DOMContentLoaded', () => {
  const translationToggle = document.getElementById('translationToggle');
  const statusText = document.getElementById('statusText');
  const targetLang = document.getElementById('targetLang');
  const apiEndpoint = document.getElementById('apiEndpoint');
  const saveButton = document.getElementById('saveSettings');
  const statusMessage = document.getElementById('statusMessage');

  // Load saved settings
  browser.storage.local.get(['enabled', 'targetLang', 'apiEndpoint']).then((result) => {
    if (result.enabled) {
      translationToggle.checked = true;
      statusText.textContent = 'Translation On';
    }
    
    if (result.targetLang) {
      targetLang.value = result.targetLang;
    }
    
    if (result.apiEndpoint) {
      apiEndpoint.value = result.apiEndpoint;
    }
  });

  // Toggle translation on/off
  translationToggle.addEventListener('change', () => {
    const enabled = translationToggle.checked;
    statusText.textContent = enabled ? 'Translation On' : 'Translation Off';
    
    // Save toggle state
    browser.storage.local.set({ enabled });
    
    // Send message to background script
    browser.runtime.sendMessage({
      action: enabled ? 'enableTranslation' : 'disableTranslation'
    });
    
    // Send message to content script on active tab
    browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
      if (tabs[0]) {
        browser.tabs.sendMessage(tabs[0].id, {
          action: enabled ? 'enableTranslation' : 'disableTranslation'
        });
      }
    });
  });

  // Save settings
  saveButton.addEventListener('click', () => {
    const settings = {
      targetLang: targetLang.value,
      apiEndpoint: apiEndpoint.value.trim()
    };
    
    // Validate API endpoint
    if (!settings.apiEndpoint) {
      statusMessage.textContent = 'Please enter a valid API endpoint';
      return;
    }
    
    // Save settings
    browser.storage.local.set(settings).then(() => {
      statusMessage.textContent = 'Settings saved successfully!';
      
      // Notify background script of settings change
      browser.runtime.sendMessage({
        action: 'settingsUpdated',
        settings
      });
      
      // Clear message after 3 seconds
      setTimeout(() => {
        statusMessage.textContent = '';
      }, 3000);
    });
  });
}); 