// --- Existing JS ---
const promptInput = document.getElementById('prompt-input');
const parametersInput = document.getElementById('parameters-input');
const generateButton = document.getElementById('generate');
const outputContainer = document.getElementById('output-container');
const loadingIndicator = document.getElementById('loading-indicator');

// --- New Lightbox Elements ---
const lightboxOverlay = document.getElementById('lightbox-overlay');
const lightboxImage = document.getElementById('lightbox-image');

// --- Function to open the lightbox ---
function openLightbox(imageUrl) {
    lightboxImage.src = imageUrl; // Set the source of the lightbox image
    lightboxOverlay.classList.remove('lightbox-hidden'); // Make overlay visible
    document.body.classList.add('lightbox-active'); // Prevent body scroll
}

// --- Function to close the lightbox ---
function closeLightbox() {
    lightboxOverlay.classList.add('lightbox-hidden'); // Hide overlay
    document.body.classList.remove('lightbox-active'); // Allow body scroll
    // Optional: Clear src after transition to prevent flash of old image if reopened quickly
    setTimeout(() => {
        // Check if it's still hidden before clearing src
        if (lightboxOverlay.classList.contains('lightbox-hidden')) {
            lightboxImage.src = "";
        }
    }, 300); // Match CSS transition duration (0.3s)
}

// --- Event Listener for clicking generated images (using Event Delegation) ---
outputContainer.addEventListener('click', (event) => {
    // Check if the clicked element is an IMG inside the output container
    // and *not* the loading indicator itself (if it were an img)
    if (event.target.tagName === 'IMG' && event.target.closest('#output-container') && !event.target.closest('#loading-indicator')) {
        openLightbox(event.target.src); // Open lightbox with the clicked image's source
    }
});

// --- Event Listener for closing the lightbox ---
// Close when clicking on the overlay background OR the image itself
lightboxOverlay.addEventListener('click', (event) => {
    // We don't need to check event.target here, any click within the overlay closes it.
    closeLightbox();
});

// --- Existing Generate Button Logic ---
generateButton.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    const parameters = parametersInput.value.trim();

    if (!prompt) {
        alert('Please enter a prompt.');
        return;
    }

    const existingError = outputContainer.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    outputContainer.insertBefore(loadingIndicator, outputContainer.firstChild);
    loadingIndicator.classList.remove('hidden');
    loadingIndicator.classList.add('loading');
    loadingIndicator.textContent = '';

    try {
        const response = await fetch('/easyrun', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, parameters: parameters }),
        });

        if (!response.ok) {
            let errorDetails = `HTTP error! Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += ` - ${errorData.message || JSON.stringify(errorData)}`;
            } catch (e) { errorDetails += ` - ${response.statusText}`; }
            throw new Error(errorDetails);
        }

        const imageUrls = await response.json();

        loadingIndicator.classList.add('hidden');
        loadingIndicator.classList.remove('loading');
        loadingIndicator.textContent = 'Loading...';

        if (Array.isArray(imageUrls) && imageUrls.length > 0) {
            const firstContentElement = outputContainer.querySelector('img, .error-message');
             imageUrls.forEach(url => {
                const imgElement = document.createElement('img');
                imgElement.src = url;
                imgElement.alt = `Generated image for prompt: ${prompt.substring(0, 50)}...`;
                imgElement.classList.add('generated-image'); // Ensure this class exists
                // **Important**: Add loading="lazy" for potentially many images
                imgElement.loading = 'lazy';
                outputContainer.insertBefore(imgElement, firstContentElement || outputContainer.firstChild);
             });
        } else {
             console.warn("Received empty array or unexpected data format:", imageUrls);
             displayErrorMessage('Received no images or unexpected data format.');
        }

    } catch (error) {
        console.error('Error generating images:', error);
        loadingIndicator.classList.add('hidden');
        loadingIndicator.classList.remove('loading');
        loadingIndicator.textContent = 'Loading...';
        displayErrorMessage(`Error: ${error.message}`);
    }
});

// --- Existing Error Message Function ---
function displayErrorMessage(message) {
    const existingError = outputContainer.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    const errorElement = document.createElement('div');
    errorElement.textContent = message;
    errorElement.style.color = 'red';
    errorElement.style.marginTop = '10px';
    // Add specific class for easier selection if needed later
    errorElement.classList.add('error-message');
    // Ensure error message is also inserted at the top
    outputContainer.insertBefore(errorElement, outputContainer.firstChild);
}

// --- Initial Setup (optional, good practice) ---
document.addEventListener('DOMContentLoaded', () => {
   // Ensure lightbox is hidden on load, even if CSS fails somehow
   if (!lightboxOverlay.classList.contains('lightbox-hidden')) {
       lightboxOverlay.classList.add('lightbox-hidden');
   }
   // Ensure loading indicator is hidden (if not already by HTML class)
   if (!loadingIndicator.classList.contains('hidden')) {
       loadingIndicator.classList.add('hidden');
   }
});