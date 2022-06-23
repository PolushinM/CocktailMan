function show() {
    const ingredients = document.getElementById('ingredients');
    if (ingredients.style.display === "none") {
       ingredients.style.display = "block";
       ingredients.scrollIntoView({block: "center", behavior: "smooth"});
       document.getElementById("ingred_embed").style.color = "#153e59";
    } else {
        document.getElementById('ingredients').style.display = "none"
       }
}