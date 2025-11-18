document.getElementById("fraudForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    let formData = new FormData(event.target);
    let jsonData = {};

    formData.forEach((value, key) => {
        jsonData[key] = parseFloat(value);
    });

    let resultBox = document.getElementById("result");
    resultBox.style.display = "none";

    try {
        const response = await fetch("http://localhost:9000/predict", {

            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(jsonData)
        });

        const data = await response.json();

        if (data.prediction === 1) {
            resultBox.className = "danger";
            resultBox.innerText = "⚠️ FRAUD TRANSACTION DETECTED!";
        } else {
            resultBox.className = "success";
            resultBox.innerText = "✔ SAFE TRANSACTION";
        }

        resultBox.style.display = "block";

    } catch (error) {
        resultBox.className = "danger";
        resultBox.innerText = "❌ Error connecting to server!";
        resultBox.style.display = "block";
    }
});
