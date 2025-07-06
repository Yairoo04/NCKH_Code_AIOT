// home.js - Client-side logic for DDoS Detection System homepage

// Constants
const STATUS_UPDATE_INTERVAL = 5000; // Update interval in milliseconds (5 seconds)
const MAX_RETRIES = 3; // Maximum number of retry attempts
const RETRY_DELAY_BASE = 1000; // Base retry delay in milliseconds (1 second)

// Global variables
let retryCount = 0;

/**
 * Updates the status display by fetching data from the /api/status endpoint
 */
function updateStatus() {
    const statusDisplay = document.getElementById('status-display');
    if (!statusDisplay) {
        console.error('Status display element not found');
        return;
    }

    // Set loading state
    statusDisplay.textContent = 'Status: Loading...';
    statusDisplay.className = 'status-normal';

    fetch('/api/status', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Reset retry count on successful fetch
        retryCount = 0;

        // Update status display
        statusDisplay.textContent = `Status: ${data.status || 'Unknown'}`;
        statusDisplay.className = data.status === 'Normal' ? 'status-normal' : 'status-attack';
    })
    .catch(error => {
        console.error('Error fetching status:', error);

        // Update display with error state
        statusDisplay.textContent = 'Status: Error';
        statusDisplay.className = 'status-error';

        // Retry logic
        if (retryCount < MAX_RETRIES) {
            retryCount++;
            const retryDelay = RETRY_DELAY_BASE * retryCount;
            console.log(`Retrying in ${retryDelay / 1000} seconds (Attempt ${retryCount}/${MAX_RETRIES})`);
            setTimeout(updateStatus, retryDelay);
        }
    });
}

// Initial status update on page load
document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    // Periodic updates
    setInterval(updateStatus, STATUS_UPDATE_INTERVAL);
});