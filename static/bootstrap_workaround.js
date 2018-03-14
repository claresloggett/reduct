
function main(){
  console.log("bootstrapfix: loaded");
  var elements = document.getElementsByClassName('data-toggle-tab');
  console.log("bootstrapfix: ",elements.length," elements");
  //console.log(elements);
  for (i=0; i<elements.length; i++){
    elements[i].setAttribute('data-toggle', 'tab');
    console.log("bootstrapfix: ",elements[i].getAttribute('href'));
    var hrefParts = elements[i].getAttribute('href').split('/');
    elements[i].setAttribute('href', hrefParts[hrefParts.length-1]);
    console.log("bootstrapfix: ",hrefParts[hrefParts.length-1]);
  }
  //console.log(elements);
}

$(document).ready(main);
