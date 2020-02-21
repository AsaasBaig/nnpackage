// initialise HTML DOM elements for reference.
const trainButton = document.getElementById('train');
const trainEpochs = document.getElementById("train-epochs");
const statusElement = document.getElementById("training-status");
const imagesElement = document.getElementById("images");

//export function = function that can be imported via other script within dir.
//get training epochs via HTML input element for CNN webpage
export function getTrainEpochs() {
    return Number.parseInt(document.getElementById('train-epochs').value);
}

//disable UI once model training button event is triggered.
export function disableUI(){
    trainButton.setAttribute("disabled", true);
    trainEpochs.setAttribute("disabled", true);
}

//set status for UI to show when data is loaded or when model is training.
export function setStatus(message){
    statusElement.innerText = message;
}


//show test results in the form of images.
export function showTestResults(batch, predictions, labels) {
    const testExamples = batch.xs.shape[0];
    imagesElement.innerHTML = '';
    //for each test example, reshape the image as a whole image and append to prediction container divs
    for (let i = 0; i < testExamples; i++) {
      const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
  
      //create div and append image per div
      const div = document.createElement('div');
      div.className = 'pred-container';
  
      const canvas = document.createElement('canvas');
      canvas.className = 'prediction-canvas';
      draw(image.flatten(), canvas);
  
      const pred = document.createElement('div');
  
      //assign label and predictions received via script.js
      const prediction = predictions[i];
      const label = labels[i];
      //if the prediction is the exact same as label, the prediction was correct.
      const correct = prediction === label;
  
      //append div classname based on correct/incorrect for css to change colour
      pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
      pred.innerText = `Pred: ${prediction}`;
  
      div.appendChild(pred);
      div.appendChild(canvas);
  
      imagesElement.appendChild(div);
    }
  }

  //draw image based on a 28x28 canvas for the images used within the model
  export function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }