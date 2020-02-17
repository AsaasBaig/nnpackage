const trainButton = document.getElementById('train')
const trainEpochs = document.getElementById("train-epochs")
const statusElement = document.getElementById("training-status")

export function getTrainEpochs() {
    return Number.parseInt(document.getElementById('train-epochs').value);
}

export function disableUI(){
    trainButton.setAttribute("disabled", true);
    trainEpochs.setAttribute("disabled", true);
}

export function setStatus(message){
    statusElement.innerText = message;
}