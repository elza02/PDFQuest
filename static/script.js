// script.js
async function sendMessage() {
    const userInput = document.getElementById("user-input").value;

    if (!userInput) {
        return; // Prevent empty messages
    }

    // Display the user's message
    const chatWindow = document.getElementById("chat-window");
    const userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = userInput;
    chatWindow.appendChild(userMessage);

    // Clear the input field
    document.getElementById("user-input").value = "";

    // Show loading indicator
    const loadingIndicator = document.getElementById("loading");
    loadingIndicator.style.display = "block";

    // Send the user input to the FastAPI backend
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: userInput })
        });

        if (response.ok) {
            const data = await response.json();

            // Display the bot's response
            const botMessage = document.createElement("div");
            botMessage.className = "bot-message";
            botMessage.textContent = data.response;
            chatWindow.appendChild(botMessage);
        } else {
            console.error("Error:", response.statusText);
        }
    } catch (error) {
        console.error("Error sending message:", error);
    } finally {
        // Hide loading indicator
        loadingIndicator.style.display = "none";

        // Scroll to the bottom of the chat window
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

// Add an event listener for the "Enter" key
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
