// Up container functionality (already working)
const containerUp = document.querySelector(".up-container");
const dragHandleUp = document.querySelector("#dragHandleUp");

let isDraggingUp = false;
let startYUp;
let initialContainerTopUp;

dragHandleUp.addEventListener("mousedown", (e) => {
    isDraggingUp = true;
    startYUp = e.clientY;
    initialContainerTopUp = containerUp.getBoundingClientRect().top;
    document.body.style.cursor = "grabbing";
    e.preventDefault(); // Prevent text selection while dragging
});

// Down container functionality (new)
const containerDown = document.querySelector(".down-container");
const dragHandleDown = document.querySelector("#dragHandleDown");

let isDraggingDown = false;
let startYDown;
let initialContainerTopDown;

dragHandleDown.addEventListener("mousedown", (e) => {
    isDraggingDown = true;
    startYDown = e.clientY;
    initialContainerTopDown = containerDown.getBoundingClientRect().top;
    document.body.style.cursor = "grabbing";
    e.preventDefault(); // Prevent text selection while dragging
});

// Combined mousemove handler
document.addEventListener("mousemove", (e) => {
    // Handle up container dragging
    if (isDraggingUp) {
        const dy = e.clientY - startYUp;
        const newTop = initialContainerTopUp + dy;
        const windowHeight = window.innerHeight;
        
   
        const clampedTop = Math.min(Math.min(newTop, windowHeight * -0.4), windowHeight * 1);
        
        containerUp.style.top = `${clampedTop}px`;
    }
    
    // Handle down container dragging
    if (isDraggingDown) {
        const dy = e.clientY - startYDown;
        const newTop = initialContainerTopDown + dy;
        const windowHeight = window.innerHeight;
        
      
        const clampedTop = Math.min(Math.max(newTop, windowHeight * 0.1), windowHeight * 1.0);
        
        containerDown.style.top = `${clampedTop}px`;
    }
});

// Combined mouseup handler
document.addEventListener("mouseup", () => {
    if (isDraggingUp) {
        isDraggingUp = false;
        document.body.style.cursor = "default";
    }
    
    if (isDraggingDown) {
        isDraggingDown = false;
        document.body.style.cursor = "default";
    }
});

// Handle window resize for both containers
window.addEventListener("resize", () => {
    const windowHeight = window.innerHeight;
    
    // Adjust up container
    const currentTopUp = parseFloat(containerUp.style.top);
    if (currentTopUp) {
        const clampedTop = Math.min(Math.max(currentTopUp, windowHeight * -0.7), windowHeight * 1.0);
        containerUp.style.top = `${clampedTop}px`;
    }
    
    // Adjust down container
    const currentTopDown = parseFloat(containerDown.style.top);
    if (currentTopDown) {
        const clampedTop = Math.min(Math.max(currentTopDown, windowHeight * 0.7), windowHeight * 1.0);
        containerDown.style.top = `${clampedTop}px`;
    }
});