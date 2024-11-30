// script.js
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevent form from reloading the page

    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) {
        alert('Please select a file!');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        // Send the file to the backend API
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to classify genre');
        }

        const result = await response.json();
        const genre = result.genre;

        // Display the result
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('genre').textContent = genre;
    } catch (error) {
        console.error(error);
        alert('An error occurred while classifying the genre.');
    }
});