const chat = document.getElementById('chat');
const promptInput = document.getElementById('prompt');
const sendBtn = document.getElementById('send-btn');
const streamContainer = document.getElementById('stream-container');
const streamedImage = document.getElementById('streamed-image');
const interruptBtn = document.getElementById('interrupt-btn');
let streamInterval = null;

function addMessage(text, isUser = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : ''}`;
    messageDiv.innerHTML = `
        <div class="message-text">${text}</div>
        <div class="timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    
    // Add click handler for user messages
    if (isUser) {
        messageDiv.classList.add('clickable-message');
        messageDiv.addEventListener('click', () => {
            const promptText = messageDiv.querySelector('.message-text').textContent;
            promptInput.value = promptText;
            promptInput.focus();
            // Trigger height adjustment
            promptInput.dispatchEvent(new Event('input'));
        });
    }
    
    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
}
async function generateImage(prompt) {
    // Show streaming preview
    streamContainer.style.display = 'block';
    interruptBtn.style.display = 'block';

    // Start streaming updates
    let comfyUrl = '';
    try {
        const urlResponse = await fetch('/url');
        const urlData = await urlResponse.json();
        comfyUrl = urlData.url.replace(/\/$/, '');
        
        const updatePreview = () => {
            streamedImage.src = `${comfyUrl}/stream/image?t=${Date.now()}`;
        };
        streamInterval = setInterval(updatePreview, 900);
        updatePreview();
    } catch (err) {
        console.error('Failed to start streaming:', err);
    }

    try {
        const response = await fetch('/gen', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ positive_prompt: prompt })
        });

        const data = await response.json();
        console.log('API Response:', data); // Debugging log

        if (Array.isArray(data.img) && data.img.length > 0) {
            console.log('Received images:', data.img);

            data.img.forEach(imageUrl => {
                if (!imageUrl) {
                    console.error('Invalid image URL:', imageUrl);
                    return;
                }

                const imgMessage = document.createElement('div');
                imgMessage.className = 'message image-message';

                const img = document.createElement('img');
                img.src = imageUrl;
                img.alt = "Generated image";
                img.loading = "lazy";

                // Debugging log for image loading
                img.onload = () => console.log(`Image loaded: ${imageUrl}`);
                img.onerror = () => console.error(`Failed to load image: ${imageUrl}`);

                // Add click handler for fullscreen
                img.addEventListener('click', () => {
                    if (!document.fullscreenElement) {
                        const container = document.createElement('div');
                        container.className = 'fullscreen';
                        const clone = img.cloneNode();
                        container.appendChild(clone);
                        document.body.appendChild(container);
                        container.onclick = () => container.remove();
                    }
                });

                imgMessage.appendChild(img);

                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = new Date().toLocaleTimeString();

                imgMessage.appendChild(timestamp);
                chat.appendChild(imgMessage);
            });
        } else {
            addMessage(`Error: ${data.message || 'Unexpected API response'}`, false);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        addMessage(`Error: ${error.message}`, false);
    } finally {
        // Clean up streaming
        clearInterval(streamInterval);
        streamContainer.style.display = 'none';
        interruptBtn.style.display = 'none';
    }
}



sendBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    addMessage(prompt, true);
    promptInput.value = '';
    await generateImage(prompt);
});



interruptBtn.addEventListener('click', async () => {
    try {
        const urlResponse = await fetch('/url');
        const urlData = await urlResponse.json();
        await fetch(`${urlData.url}/interrupt`, { method: 'POST' });
    } catch (err) {
        console.error('Failed to interrupt:', err);
    }
});

// Textarea handling
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBtn.click();
    }
});

promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = `${promptInput.scrollHeight}px`;
});
const themeToggle = document.getElementById('theme-toggle');

function setTheme(theme) {
document.documentElement.setAttribute('data-theme', theme);
localStorage.setItem('theme', theme);
themeToggle.textContent = theme === 'dark' ? '🌙' : '☀️';
}

function toggleTheme() {
const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
setTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 
          (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
setTheme(savedTheme);

themeToggle.addEventListener('click', toggleTheme);