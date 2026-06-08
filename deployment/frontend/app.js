const API_BASE_URL =
    "http://127.0.0.1:8000";

const CLASS_MAP = {
    "AK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "UNKNOWN": "Unknown"
};

/* ==========================================
   HELPER FUNCTIONS
========================================== */

function showLoading() {

    document
        .getElementById("loading")
        .classList
        .remove("hidden");

    document
        .getElementById("results")
        .classList
        .add("hidden");
}

function hideLoading() {

    document
        .getElementById("loading")
        .classList
        .add("hidden");
}

function showResults() {

    const results =
        document.getElementById("results");

    results.classList.remove("hidden");

    results.style.opacity = "0";
    results.style.transform = "translateY(20px)";

    setTimeout(() => {

        results.style.transition =
            "all 0.5s ease";

        results.style.opacity = "1";

        results.style.transform =
            "translateY(0px)";

    }, 50);

    results.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

function getConfidenceColor(confidence) {

    if (confidence >= 80)
        return "#10b981";

    if (confidence >= 60)
        return "#2563eb";

    if (confidence >= 40)
        return "#f59e0b";

    return "#ef4444";
}

/* ==========================================
   IMAGE PREVIEW
========================================== */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        const imageInput =
            document.getElementById(
                "imageInput"
            );

        imageInput.addEventListener(
            "change",
            function () {

                const file =
                    this.files[0];

                if (!file) return;

                const preview =
                    document.getElementById(
                        "uploadedImage"
                    );

                if (preview) {

                    preview.src =
                        URL.createObjectURL(file);
                }
            }
        );
    }
);

/* ==========================================
   MAIN PREDICTION FUNCTION
========================================== */

async function predictImage() {

    const fileInput =
        document.getElementById(
            "imageInput"
        );

    if (
        fileInput.files.length === 0
    ) {

        alert(
            "Please select an image first."
        );

        return;
    }

    const file =
        fileInput.files[0];

    showLoading();

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );

        if (!response.ok) {

            throw new Error(
                "Server returned an error."
            );
        }

        const data =
            await response.json();

        hideLoading();

        showResults();

        /* ==================================
           PREDICTION
        ================================== */

        document
            .getElementById(
                "prediction"
            )
            .innerText =
            CLASS_MAP[
                data.prediction
            ] || data.prediction;

        /* ==================================
           CONFIDENCE
        ================================== */

        const confidenceElement =
            document.getElementById(
                "confidence"
            );

        confidenceElement.innerText =
            data.confidence + "%";

        confidenceElement.style.color =
            getConfidenceColor(
                data.confidence
            );

        /* ==================================
           STATUS
        ================================== */

        document
            .getElementById(
                "clarity"
            )
            .innerText =
            data.clarity;

        /* ==================================
           ALTERNATIVE
        ================================== */

        document
            .getElementById(
                "alternative"
            )
            .innerText =
            CLASS_MAP[
                data.alternative
            ] || data.alternative;

        /* ==================================
           REASON
        ================================== */

        document
            .getElementById(
                "reason"
            )
            .innerText =
            data.reason;

        /* ==================================
           DELTA
        ================================== */

        document
            .getElementById(
                "delta"
            )
            .innerText =
            Number(
                data.delta
            ).toFixed(2);

        /* ==================================
           UPLOADED IMAGE
        ================================== */

        document
            .getElementById(
                "uploadedImage"
            )
            .src =
            URL.createObjectURL(file);

        /* ==================================
           HEATMAP
        ================================== */

        const heatmapImg =
            document.getElementById(
                "heatmapImage"
            );

        const placeholder =
            document.getElementById(
                "heatmapPlaceholder"
            );

        if (data.heatmap) {

            heatmapImg.style.display =
                "block";

            placeholder.classList.add(
                "hidden"
            );

            heatmapImg.src =
                API_BASE_URL +
                data.heatmap +
                "?t=" +
                new Date().getTime();

        }

        else {

            heatmapImg.style.display =
                "none";

            placeholder.classList.remove(
                "hidden"
            );
        }

        /* ==================================
           SIMILARITY SCORES
        ================================== */

        const scoreContainer =
            document.getElementById(
                "scoreContainer"
            );

        scoreContainer.innerHTML =
            "";

        const sortedScores =
            Object.entries(
                data.all_scores
            )
                .sort(
                    (a, b) =>
                        b[1] - a[1]
                );

        sortedScores.forEach(
            (
                [label, value],
                index
            ) => {

                const width =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            value
                        )
                    );

                scoreContainer.innerHTML += `

                <div class="score-row">

                    <div class="score-label">

                        ${CLASS_MAP[label] || label}
                        (${Number(value).toFixed(2)}%)

                    </div>

                    <div class="score-bar">

                        <div
                            class="score-fill"
                            id="score-${index}"
                            style="width:0%"
                        ></div>

                    </div>

                </div>

                `;
            }
        );

        /* ==================================
           ANIMATE BARS
        ================================== */

        setTimeout(() => {

            sortedScores.forEach(
                (
                    [label, value],
                    index
                ) => {

                    const bar =
                        document.getElementById(
                            `score-${index}`
                        );

                    if (bar) {

                        bar.style.width =
                            `${value}%`;
                    }
                }
            );

        }, 300);

    }

    catch (error) {

        console.error(error);

        hideLoading();

        alert(
            "Unable to connect to backend API. Please make sure the FastAPI server is running."
        );
    }
}
