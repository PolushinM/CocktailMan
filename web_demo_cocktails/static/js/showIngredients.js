const popover_link = $('[data-toggle="ingr_popover"]');
let popower_visible = false;


$(function () {
    popover_link.popover({
    content: ingredients_list,
    html: true,
    container: '#main_container',
    placement: 'bottom',
    trigger: 'manual',
    })
});

popover_link.on('hidden.bs.popover', function () {
  popower_visible = false;
});

popover_link.on('shown.bs.popover', function () {
  popower_visible = true;
});

popover_link.click(function() {

    if (popower_visible) {
         popover_link.popover('hide');
    }
    else {
        popover_link.popover('show');
    }
});

popover_link.focus(function() {
    popover_link.popover('show');
});

popover_link.blur(function() {
    popover_link.popover('hide');
});

window.addEventListener('scroll', function(event) {
    event.stopImmediatePropagation();
}, true);

window.addEventListener('click', function(event) {
    let popover_body = Array.from(document.getElementsByClassName('popover-body'))
    if (!popover_body.includes(event.target)) {
        popover_link.popover('hide');
    }
}, true);
