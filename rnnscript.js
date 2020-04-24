import * as ui from './ui.js';

const burgerButton = document.getElementsByClassName("toggle-button")[0]
burgerButton.addEventListener("click", ui.toggleBurgerMenu, true);