document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("prediction-form");
    const submitButton = document.getElementById("submit-btn");
    const resultDiv = document.getElementById("result");

    let chartInstance = null;

    // ===== Dataset Ranges (from your dataset) =====
    const RANGES = {
        rainfall: [500.02, 1999.89],
        soil_moisture: [10, 50],
        evaporation_rate: [1, 10],
        recharge_rate: [50.03, 499.96],
        well_yield: [1, 50],
        aquifer_thickness: [10, 100]
    };

    function showError(msg) {
        resultDiv.innerHTML = `<span style="color:red;">${msg}</span>`;
    }

    function validateRange(value, min, max, name) {
        if (value < min || value > max) {
            throw new Error(`${name} must be between ${min} and ${max}`);
        }
    }

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        submitButton.disabled = true;
        submitButton.innerText = "Predicting...";

        try {
            // ===== Get Inputs =====
            const rainfall = parseFloat(document.getElementById("rainfall").value);
            const soilMoisture = parseFloat(document.getElementById("soilMoisture").value);
            const evaporationRate = parseFloat(document.getElementById("evaporationRate").value);
            const rechargeRate = parseFloat(document.getElementById("rechargeRate").value);
            const wellYield = parseFloat(document.getElementById("wellYield").value);
            const aquiferThickness = parseFloat(document.getElementById("aquiferThickness").value);

            if ([rainfall, soilMoisture, evaporationRate, rechargeRate, wellYield, aquiferThickness].some(isNaN)) {
                throw new Error("Please enter valid numeric values.");
            }

            // ===== Dataset-based Range Validation =====
            validateRange(rainfall, ...RANGES.rainfall, "Rainfall");
            validateRange(soilMoisture, ...RANGES.soil_moisture, "Soil Moisture");
            validateRange(evaporationRate, ...RANGES.evaporation_rate, "Evaporation Rate");
            validateRange(rechargeRate, ...RANGES.recharge_rate, "Recharge Rate");
            validateRange(wellYield, ...RANGES.well_yield, "Well Yield");
            validateRange(aquiferThickness, ...RANGES.aquifer_thickness, "Aquifer Thickness");

            // ===== Send to Backend (MATCH backend keys EXACTLY) =====
            const inputData = {
                rainfall_mm: rainfall,
                soil_moisture: soilMoisture,
                evaporation_rate: evaporationRate,
                recharge_rate: rechargeRate,
                well_yield: wellYield,
                aquifer_thickness: aquiferThickness
            };


            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed");
            }

            const gwLevel = parseFloat(result.groundwater_level).toFixed(2);
            const status = result.status;

            let color = "#ffc107";
            let advice = "";

            if (status === "Low") {
                color = "#f44336";
                advice = "Conserve water, recharge wells, use rainwater harvesting.";
            } else if (status === "High") {
                color = "#4caf50";
                advice = "Groundwater is healthy. Maintain sustainable usage.";
            } else {
                color = "#ffc107";
                advice = "Groundwater is stable. Continue balanced usage.";
            }

            resultDiv.innerHTML = `
                <p><strong>Predicted Groundwater Level:</strong> ${gwLevel} meters</p>
                <p><strong>Status:</strong> ${status}</p>
                <p><strong>Advice:</strong> ${advice}</p>
            `;

            // ===== Chart =====
            if (chartInstance) chartInstance.destroy();

            const ctx = document.getElementById("levelChart").getContext("2d");

            chartInstance = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: ["Groundwater Level"],
                    datasets: [{
                        label: "Level (m)",
                        data: [gwLevel],
                        backgroundColor: color
                    }]
                },
                options: {
                    animation: { duration: 1200 },
                    scales: { y: { beginAtZero: true } }
                }
            });

        } catch (error) {
            showError(error.message);
        }

        submitButton.disabled = false;
        submitButton.innerText = "Predict";
    });
});
