$( function() {
    $('#make_button').click(function (e) {
        e.preventDefault()
        const formData = new FormData();
        formData.append("input_file", $('#file-selector')[0].files[0]);
        formData.append("image_url", $('#input_url').val());
        AjaxRequest(formData);
    });

    $("#input_url").focus(function() {
        document.getElementById('file-selector').value = null;
        document.getElementById('input_url').value = null;
    });

})

function AjaxRequest(formData) {
    let button = $('#make_button')
    button.prop('disabled', true)
    button.children(0).prop('hidden', false)
    console.log(button)

    $.ajax({type: 'POST',
        url: '/',
        data: formData,
        contentType: false,
        cache: false,
        processData: false,

        success: function(response) {
            let data = jQuery.parseJSON(response);
            CloseMessage();
            ShowRecipe(data);
            if (data['flash_message'] !== "") {
                FlashMessage(data['flash_message']);
                }
            button.prop('disabled', false)
            button.children(0).prop('hidden', true)
            console.log(button)
            },

        error: function(error) {
            console.log(error);
            },
    });
}

function ShowRecipe(data) {
    $('#img_cocktail').attr('src', data['image_path']);
    if (data['recipe'] !== "") {
        $('#recipe').text(data['recipe']);
    }
    else {
        $('#recipe').html("");
    }
    if (data['confidence'] !== "") {
        $('#confidence').text(data['confidence']);
    }
    else {
        $('#confidence').html("");
    }
}

function FlashMessage(message) {
    let template = $("#flash_message_template").html();
    $('#flash_div').html(template);
    $('.flash_message_text').text(message);
    document.getElementById('input_url').value = null;
    setTimeout("CloseMessage()", 2000);
}

function CloseMessage() {
    let messageNode = document.getElementById('flash_message')
    if (messageNode != null) {
        let message = bootstrap.Alert.getOrCreateInstance(messageNode)
        message.close()
    }
 }