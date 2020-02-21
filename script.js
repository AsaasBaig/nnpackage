import {MnistData} from './data.js';
import * as ui from './ui.js';

//on train button click, run classification for MNIST.
const trainButton = document.getElementById("train")
trainButton.addEventListener("click", run, true);

//set classnames globally as they represent the features within MNIST dataset.
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

async function run() {
  //disable buttons/input
  ui.disableUI();

  const data = new MnistData(); //create new MNIST data object
  const model = getModel();//assign model to var using getModel func

  //call visualisation and data calculation methods

  //print data status
  ui.setStatus("Loading data...")
  await data.load();
  //once data is loaded, await for model training
  await train(model, data);
  //once model is trained, show prediction images based on training results
  showPredictions(model, data);
  //show accuracy table and confusion matrix
  await showAccuracy(model, data);
  await showConfusion(model, data);

}

//pull data through and model to an async function.
async function train(model, data) {

  //set training status
  ui.setStatus("Training model... To cancel, refresh the page or close browser tab.")

  //metrics to be used by the tfvis fitcallback function, metrics will represent the data being output 
  //within live graph
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];

  //set container to html element and show live graph.
  const container = document.getElementById('graphs');
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  //set data sizes
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;
  //get epoch value from UI
  const EPOCHS = ui.getTrainEpochs();

  //reshape the data in current batch to follow appropriate inputs.
  //in this case, image is split as 28x28 (number of inputs), with convolutions following those inputs to then
  //assign 1 output out of the 10 features

  //do this for both training batch and testing batch
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  //fit model with the reshaped data for (number of epochs retrieved) and fitcallback to represent accuracy
  //and loss on live graph
  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: EPOCHS,
    shuffle: true,
    callbacks: fitCallbacks
  });

}

async function showPredictions(model, data) {
  //gather 100 test examples from dataset
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)

    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function getModel() {
  const model = tf.sequential();

  //set dimensions for input
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of the convolutional neural network, specify input shape.
  // Then specify parameters for the convolution operation that takes place in this layer.
  // Kernal size being set to 5x5 with 8 filters
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // last layer is a dense layer which has 10 output units, one for each
  // output feature classes (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // choose an optimiser, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const container = document.getElementById('graphs');
  tfvis.show.modelSummary(container, model)
  return model;
}

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}

//async function to run side by side of when the model is being trained.
async function showAccuracy(model, data) {
  //use tfvisor to get predictions and labels from model and append to class accuracy 
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = document.getElementById('accuracy-table');
  //use the labels and predictions to calculate how accurate the model's predictions were for a class feature.
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  //throw away labels as they will be reset by another function to show other graphs.
  labels.dispose();
}

//async function to run side by side of when the model is being trained.
async function showConfusion(model, data) {
  //get labels and predictions by predicting with the model
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  //append confusion matrix to html page
  const container = document.getElementById('confusion-matrix');
  //render confusion matrix using tensorflow visor with prediction values and class names.
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames,);

  labels.dispose();
  //set ui status.
  ui.setStatus("Training completed.")
}
