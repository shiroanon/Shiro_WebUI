document.addEventListener("DOMContentLoaded", function () {
    fetch("/get_blocks")
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("dynamic-container");

            data.forEach(section => {
                let sectionWrapper = document.createElement("div");
                sectionWrapper.classList.add("section");

                if (section.container) {
                    let sectionTitle = document.createElement("h4");
                    sectionTitle.textContent = section.container;
                    sectionWrapper.appendChild(sectionTitle);
                }

                section.elements.forEach(block => {
                    let elementWrapper = document.createElement("div");
                    elementWrapper.classList.add("input-group");

                    if (block.name) {
                        let label = document.createElement("label");
                        label.textContent = block.name;
                        label.setAttribute("for", block.id);
                        elementWrapper.appendChild(label);
                    }

                    let element;

                    if (block.type === "textarea") {
                        element = document.createElement("textarea");
                    } else if (block.type === "range") {
                        element = document.createElement("input");
                        element.type = "range";
                        element.id = block.id;
                        element.min = block.min || 0;
                        element.max = block.max || 100;
                        element.value = block.value || (block.min || 0);

                        let valueDisplay = document.createElement("span");
                        valueDisplay.classList.add("slider-value");
                        valueDisplay.textContent = element.value;

                        element.addEventListener("input", () => {
                            valueDisplay.textContent = element.value;
                        });

                        elementWrapper.appendChild(element);
                        elementWrapper.appendChild(valueDisplay);
                    } else if (block.type === "select") {
                        element = document.createElement("select");
                        block.options.forEach(optionValue => {
                            let option = document.createElement("option");
                            option.value = optionValue;
                            option.textContent = optionValue;
                            element.appendChild(option);
                        });
                    } else if (block.type === "file") {
                        element = document.createElement("div");
                        element.id = block.id;
                        element.classList.add("image-upload-box");
                        element.textContent = "Click to Upload Image";

                        let fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = "image/*";
                        fileInput.style.display = "none";

                        element.addEventListener("click", () => fileInput.click());

                        fileInput.addEventListener("change", function (event) {
                            const file = event.target.files[0];
                            if (file) {
                                const reader = new FileReader();
                                reader.onload = function (e) {
                                    let img = element.querySelector("img");
                                    if (!img) {
                                        img = document.createElement("img");
                                        element.innerHTML = "";
                                        element.appendChild(img);
                                    }
                                    img.src = e.target.result;
                                };
                                reader.readAsDataURL(file);
                            }
                        });

                        element.appendChild(fileInput);
                    } else if (block.type === "button") {
                        element = document.createElement("input");
                        element.id=block.id
                        element.type = "button";
                        element.value = block.value;
                    } else if (block.type === "wid-hei") {
                        element = document.createElement("input");
                        element.type = "range";
                        element.id = block.id;
                        element.min = block.min || 256;
                        element.max = block.max || 1280;
                        element.step = block.step || 8;
                        element.value = block.value || block.min || 256;

                        let snapValues = [512, 768, 1024];

                        let valueDisplay = document.createElement("span");
                        valueDisplay.classList.add("slider-value");
                        valueDisplay.textContent = element.value;

                        element.addEventListener("input", () => {
                            let val = parseInt(element.value);
                            let closest = snapValues.reduce((prev, curr) =>
                                Math.abs(curr - val) < Math.abs(prev - val) ? curr : prev
                            );

                            if (Math.abs(val - closest) < 10) {
                                element.value = closest;
                            }

                            valueDisplay.textContent = element.value;
                        });

                        elementWrapper.appendChild(element);
                        elementWrapper.appendChild(valueDisplay);
                    } else if (block.type === "float-range") {
                        element = document.createElement("input");
                        element.type = "range";
                        element.id = block.id;
                        element.min = block.min || 0;
                        element.max = block.max || 10;
                        element.step = block.step || 0.1;
                        element.value = block.value || block.min || 0;

                        let valueDisplay = document.createElement("span");
                        valueDisplay.classList.add("slider-value");
                        valueDisplay.textContent = parseFloat(element.value).toFixed(2);

                        element.addEventListener("input", () => {
                            valueDisplay.textContent = parseFloat(element.value).toFixed(2);
                        });

                        elementWrapper.appendChild(element);
                        elementWrapper.appendChild(valueDisplay);
                    }

                    if (element) {
                        element.id = block.id;
                        elementWrapper.appendChild(element);
                        sectionWrapper.appendChild(elementWrapper);
                    }
                });

                container.appendChild(sectionWrapper);
            });
        });
});


document.addEventListener("DOMContentLoaded", async function () {
    const outputContainer = document.querySelector(".output-container");

    try {
        const response = await fetch("/list");
        const data = await response.json();

        if (data.error) {
            outputContainer.innerHTML = `<p>${data.error}</p>`;
            return;
        }

        outputContainer.innerHTML = ""; // Clear previous content

        data.images.forEach(image => {
            const imgElement = document.createElement("img");
            imgElement.src = `/compress?name=${encodeURIComponent(image)}`;
            imgElement.alt = "Image";
            imgElement.classList.add("output-image");

            outputContainer.appendChild(imgElement);
        });
    } catch (error) {
        console.error("Error fetching images:", error);
    }
});
async function sendDataAndPopulateImages() {
    const container = document.getElementById("dynamic-container");
    const inputs = container.querySelectorAll("input, select, textarea, .image-upload-box");
    
    let data = {};

    inputs.forEach(input => {
        if (input.type === "file" || input.classList.contains("image-upload-box")) return;

        let value = input.value;

        if (input.type === "range") {
            value = input.step && input.step.includes(".") ? parseFloat(value) : parseInt(value);
        }

        data[input.id] = value;
    });

    try {
        const response = await fetch("/gener", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const imageUrls = await response.json();

        const outputContainer = document.querySelector(".img-container");
        outputContainer.innerHTML = ""; 

        imageUrls.forEach(url => {
            const imgElement = document.createElement("img");
            imgElement.src = url;
            imgElement.alt = "Generated Image";
            imgElement.classList.add("output-image");
            outputContainer.appendChild(imgElement);
        });
    } catch (error) {
        console.error("Error sending data:", error);
    }
}



document.addEventListener("click", function (event) {
    if (event.target.id === "submit") {
        console.log("Submit button clicked");
        sendDataAndPopulateImages();
    }
});
document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("positive_prompt");

    if (textarea) {
        textarea.value = localStorage.getItem("textareaValue") || "";

        textarea.addEventListener("input", function () {
            localStorage.setItem("textareaValue", textarea.value);
        });
    }
});
