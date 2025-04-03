const containerUp = document.querySelector(".up-container");
const dragHandleUp = document.querySelector("#dragHandleUp");

let isDragging = false;
let startY;
let initialContainerTop;

dragHandleUp.addEventListener("mousedown", (e) => {
    isDragging = true;
    startY = e.clientY;
    initialContainerTop = containerUp.getBoundingClientRect().top;
    document.body.style.cursor = "grabbing";
    e.preventDefault(); // Prevent text selection while dragging
});

document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    
    const dy = e.clientY - startY;
    const newTop = initialContainerTop + dy;
    const windowHeight = window.innerHeight;
    
    // Keep container within 10% to 70% of viewport height
    const clampedTop = Math.min(Math.max(newTop, windowHeight * -0.8), windowHeight * 0.5);
    
    containerUp.style.top = `${clampedTop}px`;
});

document.addEventListener("mouseup", () => {
    if (!isDragging) return;
    isDragging = false;
    document.body.style.cursor = "default";
});


window.addEventListener("resize", () => {
    const windowHeight = window.innerHeight;
    const currentTop = parseFloat(containerUp.style.top);
    
    if (currentTop) {
        const clampedTop = Math.min(Math.max(currentTop, windowHeight * -0.8), windowHeight * 0.5);
        containerUp.style.top = `${clampedTop}px`;
    }
});