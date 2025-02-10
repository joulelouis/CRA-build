document.addEventListener("DOMContentLoaded", function () {
    // Select all sidebar navigation links
    const navLinks = document.querySelectorAll(".sidebar .nav-link");

    // Get the last active page from localStorage
    const activePage = localStorage.getItem("activeSidebarLink");

    // Set active class to the last active link
    if (activePage) {
        navLinks.forEach(link => {
            if (link.getAttribute("href") === activePage) {
                link.classList.add("active");
            }
        });
    }

    navLinks.forEach(link => {
        link.addEventListener("click", function (event) {
            // Remove 'active' class from all links
            navLinks.forEach(nav => {
                nav.classList.remove("active");
            });

            // Add 'active' class to the clicked link
            this.classList.add("active");

            // Store active link in localStorage
            localStorage.setItem("activeSidebarLink", this.getAttribute("href"));

            // Log the clicked link to the console
            console.log(`Clicked on: ${this.innerText.trim()} | Href: ${this.getAttribute("href")}`);
        });
    });
});
