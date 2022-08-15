function generate_image() {
    let button = $('#draw_button')
    button.prop('disabled', true)
    button.children(0).prop('hidden', false)
    $.ajax({
        type: "POST",
        data: $('#ingredients_form').serialize(),
        success: function(response) {
            let image_path = jQuery.parseJSON(response)['image_path']
            $('#img_cocktail').attr('src', image_path)
            button.prop('disabled', false)
            button.children(0).prop('hidden', true)
        },
        error: function(error) {
            console.log(error);
        }
    });
}
