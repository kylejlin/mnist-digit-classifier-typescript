# MNIST Digit Classifier

A simple neural network written in TypeScript.

I built this solely for my own education, not for practical use.
As a result, this project doesn't use any neural network or math libraries, and is also not optimized for performance.

I tried to implement my neural network as closely as possible to the implementation described in [Chapter 1 of Michael Nielsen's _Neural Networks and Deep Learning_](http://neuralnetworksanddeeplearning.com/chap1.html).

The network may appear complicated because there are a lot of files in `src`, but all the business logic can be found in `network.ts` and `matrix.ts`.
All the other files are for the user interface.

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## How to use the web app

1. Open [https://kylejlin.github.io/mnist-digit-classifier-typescript](https://kylejlin.github.io/mnist-digit-classifier-typescript).

2. Create a neural network:

   a. Optional: Configure how many hidden layers you want and how many neurons you want in each layer.

   b. Click "Create network"

3. Train your neural network:

   a. Click "Train" to open the training menu.

   b. Optional: Configure the hyperparameters to your liking.

   c. Click "Start" to start the training.

   d. Wait. Training the network will probably take quite a while, especially if your computer isn't very fast or you're on mobile. Keep in mind that this code was not optimized for speed.

   e. Eventually, under the "Logs" header, you will see "Epoch 0: [some number] / 10000". At this point, you can either wait for the network to complete more training epochs, or you can tell the network to stop training after the next epoch is completed.

   f. When you want to stop training, click "Stop training after the current epoch".

   g. Wait some more. After a little while, you will be taken back to the main menu. This will probably take quite a while.

4. View your neural network in action:

   a. Click "View"

   b. Use the "Previous" and "Next" buttons to change the classified image.

   c. If you would like, you can upload and crop your own images.

5. Your network and uploaded testing images should be saved in your browser, so you shouldn't have to retrain the network and reupload the images when you revisit this page in the future.

## Data set

The dataset is from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

## License

MIT

Copyright (c) 2020 Kyle Lin
