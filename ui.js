const trainButton = document.getElementById('train');
const trainEpochs = document.getElementById("train-epochs");
const statusElement = document.getElementById("training-status");
const imagesElement = document.getElementById("images");

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

export function showTestResults(batch, predictions, labels) {
    const testExamples = batch.xs.shape[0];
    imagesElement.innerHTML = '';

    for (let i = 0; i < testExamples; i++) {
      const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
  
      const div = document.createElement('div');
      div.className = 'pred-container';
  
      const canvas = document.createElement('canvas');
      canvas.className = 'prediction-canvas';
      draw(image.flatten(), canvas);
  
      const pred = document.createElement('div');
  
      const prediction = predictions[i];
      const label = labels[i];
      const correct = prediction === label;
  
      pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
      pred.innerText = `pred: ${prediction}`;
  
      div.appendChild(pred);
      div.appendChild(canvas);
  
      imagesElement.appendChild(div);
    }
  }

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