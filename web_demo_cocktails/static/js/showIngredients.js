function show() {
    const ingredients = document.getElementById('ingredients');
    if (ingredients.style.display === "none") {
       ingredients.style.display = "block";
       ingredients.scrollIntoView({block: "center", behavior: "smooth"});
    } else {
        document.getElementById('ingredients').style.display = "none"
       }
}