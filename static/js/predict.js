let selectedModel = 'random_forest';

// Initialize model selection
document.addEventListener('DOMContentLoaded', function() {
    const modelCards = document.querySelectorAll('.model-card');
    
    // Set first model as selected by default
    if (modelCards.length > 0) {
        modelCards[0].classList.add('selected');
        selectedModel = modelCards[0].dataset.model;
    }
    
    // Handle model card clicks
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            modelCards.forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
            selectedModel = this.dataset.model;
            document.getElementById('selectedModel').value = selectedModel;
        });
    });
    
    // Handle form submission
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
});

// Update range slider values
function updateRangeValue(name, value) {
    const valueSpan = document.getElementById(name + 'Value');
    if (valueSpan) {
        // Format value based on type
        if (name === 'sleep' || name === 'activity') {
            valueSpan.textContent = parseFloat(value).toFixed(1);
        } else {
            valueSpan.textContent = Math.round(value);
        }
    }
}

// Handle prediction form submission
async function handlePrediction(e) {
    e.preventDefault();
    
    // Hide results and show loading
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('loadingSpinner').classList.remove('hidden');
    
    // Scroll to loading spinner
    document.getElementById('loadingSpinner').scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Collect form data
    const formData = {
        model: selectedModel,
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        occupation_encoded: document.getElementById('occupation').value,
        sleep_hours: document.getElementById('sleep_hours').value,
        physical_activity: document.getElementById('physical_activity').value,
        caffeine_intake: document.getElementById('caffeine_intake').value,
        alcohol_consumption: document.getElementById('alcohol_consumption').value,
        smoking: document.getElementById('smoking').value,
        family_history: document.getElementById('family_history').value,
        stress_level: document.getElementById('stress_level').value,
        heart_rate: document.getElementById('heart_rate').value,
        breathing_rate: document.getElementById('breathing_rate').value,
        sweating_level: document.getElementById('sweating_level').value,
        dizziness: document.getElementById('dizziness').value,
        medication: document.getElementById('medication').value,
        therapy_sessions: document.getElementById('therapy_sessions').value,
        recent_life_event: document.getElementById('recent_life_event').value,
        diet_quality: document.getElementById('diet_quality').value
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction. Please check your inputs and try again.');
    } finally {
        document.getElementById('loadingSpinner').classList.add('hidden');
    }
}

// Display prediction results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const predictionValue = document.getElementById('predictionValue');
    const categoryBadge = document.getElementById('categoryBadge');
    const resultDescription = document.getElementById('resultDescription');
    const modelUsed = document.getElementById('modelUsed');
    const confidenceInfo = document.getElementById('confidenceInfo');
    const confidenceValue = document.getElementById('confidenceValue');
    
    // Set prediction value
    predictionValue.textContent = result.prediction;
    
    // Set category badge
    categoryBadge.textContent = result.category + ' Anxiety';
    categoryBadge.style.backgroundColor = result.color + '20';
    categoryBadge.style.color = result.color;
    categoryBadge.style.border = `2px solid ${result.color}`;
    
    // Set description based on category
    let description = '';
    if (result.category === 'Low') {
        description = 'Your anxiety level is in the low range. This suggests relatively good management of social anxiety symptoms. Continue maintaining healthy lifestyle habits and coping strategies.';
    } else if (result.category === 'Moderate') {
        description = 'Your anxiety level is moderate. This indicates some challenges with social anxiety that may benefit from lifestyle adjustments, stress management techniques, or professional support.';
    } else {
        description = 'Your anxiety level is in the high range. This suggests significant social anxiety symptoms. Consider seeking professional support from a mental health provider for proper assessment and treatment options.';
    }
    resultDescription.textContent = description;
    
    // Set model used
    const modelNames = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'gradient_boosting': 'Gradient Boosting',
        'svr': 'Support Vector Regression',
        'neural_network': 'Neural Network'
    };
    modelUsed.textContent = modelNames[result.model_used] || result.model_used;
    
    // Set confidence if available
    if (result.confidence) {
        confidenceValue.textContent = result.confidence;
        confidenceInfo.classList.remove('hidden');
    } else {
        confidenceInfo.classList.add('hidden');
    }
    
    // Show results with animation
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Add pulse animation to prediction value
    predictionValue.style.animation = 'none';
    setTimeout(() => {
        predictionValue.style.animation = 'pulse 0.5s ease-in-out';
    }, 10);
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
`;
document.head.appendChild(style);
