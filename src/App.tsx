import React from "react";
import { option, result } from "rusty-ts";
import "./App.css";
import { AccuracyRate, LabeledImage } from "./data";
import { MnistData, mnistProm } from "./data/mnist";
import { Matrix } from "./matrix";
import { Network } from "./network";
import { testNetwork, trainNetwork } from "./networkServices";
import {
  AppState,
  CreateNetworkState,
  CropState,
  CustomImage,
  HyperParameterMenuState,
  NetworkMainMenuState,
  StateMap,
  StateType,
  TestState,
  TrainingInProgressState,
  ViewState,
} from "./state";
import { imageSaver, networkSaver } from "./stateSavers";
import { StochasticGradientDescentHyperParameters } from "./workerMessages";

export default class App extends React.Component<{}, AppState> {
  private viewImageCanvasRef: React.RefObject<HTMLCanvasElement>;

  constructor(props: {}) {
    super(props);

    this.state = getInitialState();

    this.viewImageCanvasRef = React.createRef();

    this.bindMethods();

    (window as any).app = this;
  }

  bindMethods(): void {
    this.onCreateNetworkClick = this.onCreateNetworkClick.bind(this);
    this.onTrainClick = this.onTrainClick.bind(this);
    this.onTestClick = this.onTestClick.bind(this);
    this.onViewClick = this.onViewClick.bind(this);
    this.onResetClick = this.onResetClick.bind(this);
    this.onBatchSizeInputValueChange = this.onBatchSizeInputValueChange.bind(
      this
    );
    this.onEpochsInputValueChange = this.onEpochsInputValueChange.bind(this);
    this.onLearningRateInputValueChange = this.onLearningRateInputValueChange.bind(
      this
    );
    this.onStartTrainingClick = this.onStartTrainingClick.bind(this);
    this.onEpochComplete = this.onEpochComplete.bind(this);
    this.onTrainerTerminate = this.onTrainerTerminate.bind(this);
    this.onExitHyperParameterMenuClick = this.onExitHyperParameterMenuClick.bind(
      this
    );
    this.onStopTrainingClick = this.onStopTrainingClick.bind(this);
    this.onTestMenuExitClick = this.onTestMenuExitClick.bind(this);
    this.onExitViewMenuClick = this.onExitViewMenuClick.bind(this);
    this.onPreviousImageClick = this.onPreviousImageClick.bind(this);
    this.onNextImageClick = this.onNextImageClick.bind(this);
  }

  componentDidMount(): void {
    mnistProm.then((mnist) => {
      this.setState({ mnist: option.some(mnist) });
    });
  }

  saveState(state: AppState): void {
    this.setState(state);

    if ("network" in state) {
      const { network } = state;
      networkSaver.saveState(network);
    }
  }

  render(): React.ReactElement {
    const { state } = this;
    switch (state.stateType) {
      case StateType.CreateNetwork:
        return this.renderCreateNetworkMenu(state);
      case StateType.NetworkMainMenu:
        return this.renderNetworkMainMenu(state);
      case StateType.HyperParameterMenu:
        return this.renderHyperParameterMenu(state);
      case StateType.TrainingInProgress:
        return this.renderTrainingInProgressMenu(state);
      case StateType.Test:
        return this.renderTestMenu(state);
      case StateType.View:
        return this.renderViewMenu(state);
      case StateType.Crop:
        return this.renderCropMenu(state);
    }
  }

  renderCreateNetworkMenu(state: CreateNetworkState): React.ReactElement {
    return (
      <div className="App">
        <h1>
          {state.previousNetwork.match({
            none: () => "Create network",
            some: () => "Reset network",
          })}
        </h1>

        {state.previousNetwork.match({
          none: () => null,
          some: (network) => (
            <button onClick={() => this.cancelNetworkReset(network)}>
              Cancel
            </button>
          ),
        })}

        <h2>Layers:</h2>
        <ol>
          <li>
            Input layer: 784 neurons
            <button onClick={() => this.addLayerBelow(0, state)}>
              Add layer below
            </button>
          </li>

          {state.hiddenLayerSizeInputValues.map((value, hiddenLayerIndex) => {
            const layerIndex = hiddenLayerIndex + 1;
            return (
              <li key={layerIndex}>
                <label>
                  Hidden layer:{" "}
                  <input
                    type="text"
                    className={isPositiveIntStr(value) ? "" : "InvalidInput"}
                    value={value}
                    onChange={(e) =>
                      this.changeLayerSizeValue(
                        layerIndex,
                        e.target.value,
                        state
                      )
                    }
                  />{" "}
                  neurons
                </label>
                <button onClick={() => this.addLayerBelow(layerIndex, state)}>
                  Add layer below
                </button>
                <button onClick={() => this.deleteLayer(layerIndex, state)}>
                  Delete layer
                </button>
              </li>
            );
          })}

          <li>Output layer: 10 neurons</li>
        </ol>

        <button
          onClick={this.onCreateNetworkClick}
          disabled={state.hiddenLayerSizeInputValues.some(
            (value) => !isPositiveIntStr(value)
          )}
        >
          Create network
        </button>
      </div>
    );
  }

  renderNetworkMainMenu(state: NetworkMainMenuState): React.ReactElement {
    return (
      <div className="App">
        <h1>Explore your neural network</h1>
        <h2>Neurons in each layer: {state.network.sizes.join(", ")}</h2>

        <button onClick={this.onTrainClick}>Train</button>
        <button onClick={this.onTestClick}>Test</button>
        <button onClick={this.onViewClick}>View</button>
        <button onClick={this.onResetClick}>Reset</button>
      </div>
    );
  }

  renderHyperParameterMenu(state: HyperParameterMenuState): React.ReactElement {
    return (
      <div className="App">
        <h1>Train (stochastic gradient descent)</h1>

        <button onClick={this.onExitHyperParameterMenuClick}>Cancel</button>

        <h2>Hyperparameters:</h2>

        <label>
          Batch size:{" "}
          <input
            type="text"
            className={
              isPositiveIntStr(state.batchSizeInputValue) ? "" : "InvalidInput"
            }
            value={state.batchSizeInputValue}
            onChange={this.onBatchSizeInputValueChange}
          />
        </label>

        <label>
          Epochs:{" "}
          <input
            type="text"
            className={
              isPositiveIntStr(state.epochsInputValue) ? "" : "InvalidInput"
            }
            value={state.epochsInputValue}
            onChange={this.onEpochsInputValueChange}
          />
        </label>

        <label>
          Learning rate:{" "}
          <input
            type="text"
            className={
              isPositiveNumStr(state.learningRateInputValue)
                ? ""
                : "InvalidInput"
            }
            value={state.learningRateInputValue}
            onChange={this.onLearningRateInputValueChange}
          />
        </label>

        <button
          onClick={this.onStartTrainingClick}
          disabled={
            !(
              isPositiveIntStr(state.batchSizeInputValue) &&
              isPositiveIntStr(state.epochsInputValue) &&
              isPositiveNumStr(state.learningRateInputValue)
            )
          }
        >
          Start
        </button>
      </div>
    );
  }

  renderTrainingInProgressMenu(
    state: TrainingInProgressState
  ): React.ReactElement {
    return (
      <div className="App">
        <h1>Training in progress...</h1>

        <h2>Logs:</h2>
        {state.epochAccuracyRates
          .slice()
          .sort((a, b) => a.epoch - b.epoch)
          .map((rate) => (
            <div key={rate.epoch}>
              Epoch {rate.epoch}: {rate.correct} / {rate.total}
            </div>
          ))}

        <button onClick={this.onStopTrainingClick}>
          Stop training after current epoch
        </button>
      </div>
    );
  }

  renderTestMenu(state: TestState): React.ReactElement {
    return (
      <div className="App">
        <h1>Test results:</h1>

        {state.accuracyRate.match({
          err: () => <p>Running tests...</p>,
          ok: (rate) => (
            <p>
              {rate.correct} / {rate.total} correct
            </p>
          ),
        })}

        <button onClick={this.onTestMenuExitClick}>Back</button>
      </div>
    );
  }

  renderViewMenu(state: ViewState): React.ReactElement {
    return state.mnist.match({
      none: () => (
        <div className="App">
          <p>Loading...</p>
        </div>
      ),
      some: (mnist) => {
        const viewedImage: LabeledImage = getViewedImage(state, mnist);
        const guess = guessDigit(state.network, viewedImage.inputs);

        return (
          <div className="App">
            <h1>View classifications</h1>
            <button onClick={this.onExitViewMenuClick}>Back</button>

            <div>
              <button onClick={this.onPreviousImageClick}>Previous</button>{" "}
              Image {state.viewedIndex + 1} /{" "}
              {mnist.test.length + state.customImages.length}{" "}
              <button onClick={this.onNextImageClick}>Next</button>
            </div>
            <canvas ref={this.viewImageCanvasRef}></canvas>
            <div
              className={
                guess.digit === viewedImage.label ? "" : "IncorrectGuess"
              }
            >
              Guess: {guess.digit} ({(guess.confidence * 100).toFixed(2)}%
              confident)
            </div>
            <div>Actual: {viewedImage.label}</div>
          </div>
        );
      },
    });
  }

  renderCropMenu(state: CropState): React.ReactElement {
    return (
      <div className="App">
        <p>TODO crop menu</p>
      </div>
    );
  }

  componentDidUpdate(): void {
    const { state } = this;
    state.mnist.ifSome((mnist) => {
      if (state.stateType === StateType.View) {
        const canvas = this.viewImageCanvasRef.current;
        if (canvas !== null) {
          const viewedImage = getViewedImage(state, mnist);
          paintImage(viewedImage, canvas);
        }
      }
    });
  }

  cancelNetworkReset(network: Network): void {
    const newState: NetworkMainMenuState = {
      mnist: this.state.mnist,

      stateType: StateType.NetworkMainMenu,

      network,
    };
    this.saveState(newState);
  }

  changeLayerSizeValue(
    changedLayerIndex: number,
    newValue: string,
    state: CreateNetworkState
  ): void {
    this.saveState({
      ...state,
      hiddenLayerSizeInputValues: state.hiddenLayerSizeInputValues.map(
        (value, hiddenLayerIndex) => {
          const layerIndex = hiddenLayerIndex + 1;
          if (layerIndex === changedLayerIndex) {
            return newValue;
          } else {
            return value;
          }
        }
      ),
    });
  }

  addLayerBelow(layerIndex: number, state: CreateNetworkState): void {
    this.saveState({
      ...state,
      hiddenLayerSizeInputValues: state.hiddenLayerSizeInputValues
        .slice(0, layerIndex)
        .concat(["16"], state.hiddenLayerSizeInputValues.slice(layerIndex)),
    });
  }

  deleteLayer(layerIndex: number, state: CreateNetworkState): void {
    const hiddenLayerIndex = layerIndex - 1;
    this.saveState({
      ...state,
      hiddenLayerSizeInputValues: state.hiddenLayerSizeInputValues
        .slice(0, hiddenLayerIndex)
        .concat(state.hiddenLayerSizeInputValues.slice(hiddenLayerIndex + 1)),
    });
  }

  onCreateNetworkClick(): void {
    const state = this.expectState(StateType.CreateNetwork);
    if (state.hiddenLayerSizeInputValues.every(isPositiveIntStr)) {
      const hiddenLayerSizes: number[] = state.hiddenLayerSizeInputValues.map(
        (str) => +str
      );
      const layerSizes = [784, ...hiddenLayerSizes, 10];

      const newState: NetworkMainMenuState = {
        mnist: this.state.mnist,

        stateType: StateType.NetworkMainMenu,

        network: new Network(layerSizes),
      };

      this.saveState(newState);
    }
  }

  expectState<T extends StateType>(stateType: T): StateMap[T] {
    const { state } = this;
    if (state.stateType === stateType) {
      return state as StateMap[T];
    }
    throw new Error(
      "Expecting a state of type " +
        StateType[stateType] +
        " but got state of type " +
        StateType[state.stateType]
    );
  }

  onTrainClick(): void {
    const state = this.expectState(StateType.NetworkMainMenu);
    const newState: HyperParameterMenuState = {
      mnist: state.mnist,

      stateType: StateType.HyperParameterMenu,

      network: state.network,

      batchSizeInputValue: "10",
      epochsInputValue: "30",
      learningRateInputValue: "3.0",
    };
    this.saveState(newState);
  }

  onTestClick(): void {
    const state = this.expectState(StateType.NetworkMainMenu);

    const networkTester = testNetwork(state.network, {
      onComplete: (accuracyRate) => {
        this.updateState(StateType.Test, {
          accuracyRate: result.ok(accuracyRate),
        });
      },
    });

    const newState: TestState = {
      mnist: state.mnist,

      stateType: StateType.Test,

      network: state.network,

      accuracyRate: result.err(networkTester),
    };

    this.saveState(newState);

    networkTester.start();
  }

  updateState<T extends StateType>(
    stateType: T,
    updateOrUpdater:
      | Partial<StateMap[T]>
      | ((prevState: StateMap[T]) => Partial<StateMap[T]>)
  ): void {
    const { state } = this;
    if (state.stateType === stateType) {
      if ("function" === typeof updateOrUpdater) {
        this.saveState({ ...state, ...updateOrUpdater(state as StateMap[T]) });
      } else {
        this.saveState({ ...state, ...updateOrUpdater });
      }
    }
  }

  onViewClick(): void {
    const state = this.expectState(StateType.NetworkMainMenu);
    const newState: ViewState = {
      mnist: state.mnist,

      stateType: StateType.View,

      network: state.network,

      viewedIndex: 0,
      customImages: imageSaver.getState().unwrapOr([]),
    };
    this.saveState(newState);
  }

  onResetClick(): void {
    const state = this.expectState(StateType.NetworkMainMenu);
    const newState: CreateNetworkState = {
      mnist: state.mnist,

      stateType: StateType.CreateNetwork,

      hiddenLayerSizeInputValues: ["30"],
      previousNetwork: option.some(state.network),
    };
    this.saveState(newState);
  }

  onBatchSizeInputValueChange(
    event: React.ChangeEvent<HTMLInputElement>
  ): void {
    const state = this.expectState(StateType.HyperParameterMenu);
    const newState: HyperParameterMenuState = {
      ...state,
      batchSizeInputValue: event.target.value,
    };
    this.saveState(newState);
  }

  onEpochsInputValueChange(event: React.ChangeEvent<HTMLInputElement>): void {
    const state = this.expectState(StateType.HyperParameterMenu);
    const newState: HyperParameterMenuState = {
      ...state,
      epochsInputValue: event.target.value,
    };
    this.saveState(newState);
  }

  onLearningRateInputValueChange(
    event: React.ChangeEvent<HTMLInputElement>
  ): void {
    const state = this.expectState(StateType.HyperParameterMenu);
    const newState: HyperParameterMenuState = {
      ...state,
      learningRateInputValue: event.target.value,
    };
    this.saveState(newState);
  }

  onStartTrainingClick(): void {
    const state = this.expectState(StateType.HyperParameterMenu);

    const hyperParams: StochasticGradientDescentHyperParameters = {
      batchSize: +state.batchSizeInputValue,
      epochs: +state.epochsInputValue,
      learningRate: +state.learningRateInputValue,
    };

    const networkTrainer = trainNetwork(state.network, hyperParams, {
      onEpochComplete: this.onEpochComplete,

      onTerminate: this.onTrainerTerminate,
    });

    const newState: TrainingInProgressState = {
      mnist: state.mnist,

      stateType: StateType.TrainingInProgress,

      network: state.network,

      networkTrainer,
      epochAccuracyRates: [],
    };

    this.saveState(newState);

    networkTrainer.start();
  }

  onEpochComplete(accuracyRate: AccuracyRate, epoch: number): void {
    this.updateState(StateType.TrainingInProgress, (prevState) => ({
      epochAccuracyRates: prevState.epochAccuracyRates.concat([
        { ...accuracyRate, epoch },
      ]),
    }));
  }

  onTrainerTerminate(updatedNetwork: Network): void {
    const state = this.expectState(StateType.TrainingInProgress);
    const newState: NetworkMainMenuState = {
      mnist: state.mnist,

      stateType: StateType.NetworkMainMenu,

      network: updatedNetwork,
    };
    this.saveState(newState);
  }

  onExitHyperParameterMenuClick(): void {
    const state = this.expectState(StateType.HyperParameterMenu);
    const newState: NetworkMainMenuState = {
      mnist: state.mnist,

      stateType: StateType.NetworkMainMenu,

      network: state.network,
    };
    this.saveState(newState);
  }

  onStopTrainingClick(): void {
    const state = this.expectState(StateType.TrainingInProgress);
    state.networkTrainer.terminate();
  }

  onTestMenuExitClick(): void {
    const state = this.expectState(StateType.Test);

    state.accuracyRate.ifErr((tester) => {
      tester.terminate();
    });

    const newState: NetworkMainMenuState = {
      mnist: state.mnist,

      stateType: StateType.NetworkMainMenu,

      network: state.network,
    };
    this.saveState(newState);
  }

  onExitViewMenuClick(): void {
    const state = this.expectState(StateType.View);

    const newState: NetworkMainMenuState = {
      mnist: state.mnist,

      stateType: StateType.NetworkMainMenu,

      network: state.network,
    };
    this.saveState(newState);
  }

  onPreviousImageClick(): void {
    const state = this.expectState(StateType.View);
    state.mnist.ifSome((mnist) => {
      const numberOfImages = mnist.test.length + state.customImages.length;
      const newIndex =
        state.viewedIndex === 0 ? numberOfImages - 1 : state.viewedIndex - 1;
      this.setState({ ...state, viewedIndex: newIndex });
    });
  }

  onNextImageClick(): void {
    const state = this.expectState(StateType.View);
    state.mnist.ifSome((mnist) => {
      const numberOfImages = mnist.test.length + state.customImages.length;
      const newIndex =
        state.viewedIndex === numberOfImages - 1 ? 0 : state.viewedIndex + 1;
      this.setState({ ...state, viewedIndex: newIndex });
    });
  }
}

function getInitialState(): AppState {
  return networkSaver.getState().match({
    none: (): CreateNetworkState => ({
      mnist: option.none(),

      stateType: StateType.CreateNetwork,

      hiddenLayerSizeInputValues: ["30"],
      previousNetwork: option.none(),
    }),

    some: (network): NetworkMainMenuState => ({
      mnist: option.none(),

      stateType: StateType.NetworkMainMenu,

      network,
    }),
  });
}

function isPositiveIntStr(s: string): boolean {
  return Number.isFinite(+s) && +s === Math.floor(+s) && +s > 0;
}

function isPositiveNumStr(s: string): boolean {
  return Number.isFinite(+s) && +s > 0;
}

function getViewedImage(state: ViewState, mnist: MnistData): LabeledImage {
  return state.viewedIndex < mnist.test.length
    ? mnist.test[state.viewedIndex]
    : normalizeU8Image(
        state.customImages[state.viewedIndex - mnist.test.length]
      );
}

function normalizeU8Image(image: CustomImage): LabeledImage {
  return {
    rows: 28,
    columns: 28,
    inputs: image.u8Matrix.immutApplyElementwise((x) => x / 255),
    label: image.label,
  };
}

function guessDigit(
  network: Network,
  inputs: Matrix
): { digit: number; confidence: number } {
  const { activations } = network.performForwardPass(inputs);
  const outputActivations = activations[
    activations.length - 1
  ].rowMajorOrderEntries();

  let maxIndex = 0;
  let maxConfidence = outputActivations[maxIndex];
  for (let i = 1; i < outputActivations.length; i++) {
    const confidence = outputActivations[i];
    if (confidence > maxConfidence) {
      maxConfidence = confidence;
      maxIndex = i;
    }
  }

  return { digit: maxIndex, confidence: maxConfidence };
}

function paintImage(image: LabeledImage, canvas: HTMLCanvasElement): void {
  canvas.width = image.columns;
  canvas.height = image.rows;

  const ctx = canvas.getContext("2d")!;
  const imageData = getImageData(image);
  ctx.putImageData(imageData, 0, 0);
}

function getImageData(image: LabeledImage): ImageData {
  const entries = image.inputs.rowMajorOrderEntries();
  const imageBytes = new Uint8ClampedArray(entries.length * 4);
  for (let i = 0; i < entries.length; i++) {
    const lightness = 255 - Math.floor(entries[i] * 255);
    imageBytes[i * 4] = lightness;
    imageBytes[i * 4 + 1] = lightness;
    imageBytes[i * 4 + 2] = lightness;
    imageBytes[i * 4 + 3] = 255;
  }
  return new ImageData(imageBytes, image.columns, image.rows);
}
