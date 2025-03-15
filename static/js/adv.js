document.addEventListener("DOMContentLoaded", function () {
    fetch("/get_blocks")
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("dynamic-container");

            data.forEach(block => {
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

                    fileInput.addEventListener("change", function(event) {
                        const file = event.target.files[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = function(e) {
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
                    element.type = "button";
                    element.value = block.value;
                }

                if (element) {
                    element.id = block.id;
                    elementWrapper.appendChild(element);
                    container.appendChild(elementWrapper);
                }
            });
        });
});