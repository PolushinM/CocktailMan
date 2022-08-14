function generate_image() {
    $.ajax({
        type: "POST",
        data: $('#ingredients_form').serialize(),
        success: function(response) {
            let image_path = jQuery.parseJSON(response)['image_path']
            $('#img_cocktail').attr('src', image_path)
        },
        error: function(error) {
            console.log(error);
        }
    });
}

