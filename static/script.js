document.getElementById('submit-button').addEventListener('click', function () {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim()) {
        // Add user message to chat history
        addToChatHistory("You", userInput);

        // Clear input field
        document.getElementById('user-input').value = '';

        // Send the question to the backend
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: userInput }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                data.response.forEach(chunk => {
                    addToChatHistory("Bot", chunk);
                });
            } else if (data.error) {
                addToChatHistory("Error", data.error);
            }
        })
        .catch(error => {
            addToChatHistory("Error", "Something went wrong.");
            console.error('Error:', error);
        });
    } else {
        alert("Please enter a question!");
    }
});

function addToChatHistory(role, message) {
    const chatHistory = document.getElementById('chat-history');
    const newMessage = document.createElement('div');
    newMessage.innerHTML = `<strong>${role}:</strong> ${message}`;
    chatHistory.appendChild(newMessage);
}
