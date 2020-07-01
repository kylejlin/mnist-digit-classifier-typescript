import React from "react";
import { option, Option, result } from "rusty-ts";
import "./App.css";
import { AccuracyRate, LabeledImage } from "./data";
import { MnistData, mnistProm } from "./data/mnist";
import { Matrix } from "./matrix";
import {
  Network,
  StochasticGradientDescentHyperParameters,
  WeightInitializationMethod,
} from "./network";
import { networkFactory } from "./network/networkFactory";
import { testNetwork, trainNetwork } from "./networkServices";
import {
  AppState,
  Corner,
  CreateNetworkState,
  CropState,
  CustomImage,
  Draggable,
  HyperParameterMenuState,
  MnistLoadingErrorState,
  NetworkMainMenuState,
  Square,
  SquareAdjustment,
  StateMap,
  StateType,
  TestState,
  TrainingInProgressState,
  ViewState,
} from "./state";
import { imageSaver, networkSaver } from "./stateSavers";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface CornerAdjustment extends SquareAdjustment {
  dragged: Corner;
}

export default class App extends React.Component<{}, AppState> {
  private viewImageCanvasRef: React.RefObject<HTMLCanvasElement>;
  private customImageInputRef: React.RefObject<HTMLInputElement>;
  private cropImageCanvasRef: React.RefObject<HTMLCanvasElement>;

  constructor(props: {}) {
    super(props);

    this.state = getInitialState();

    this.viewImageCanvasRef = React.createRef();
    this.customImageInputRef = React.createRef();
    this.cropImageCanvasRef = React.createRef();

    this.bindMethods();

    (window as any).app = this;
  }

  bindMethods(): void {
    this.onWeightInitializationMethodChange = this.onWeightInitializationMethodChange.bind(
      this
    );
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
    this.onRegularizationRateInputValueChange = this.onRegularizationRateInputValueChange.bind(
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
    this.onCustomImageInputChange = this.onCustomImageInputChange.bind(this);
    this.enterCropMenu = this.enterCropMenu.bind(this);
    this.onCropImageCanvasPointerDown = this.onCropImageCanvasPointerDown.bind(
      this
    );
    this.onCropImageCanvasPointerMove = this.onCropImageCanvasPointerMove.bind(
      this
    );
    this.onCropImageCanvasPointerUp = this.onCropImageCanvasPointerUp.bind(
      this
    );
    this.onShouldInvertInputChange = this.onShouldInvertInputChange.bind(this);
    this.onDarknessThresholdChange = this.onDarknessThresholdChange.bind(this);
    this.onCustomImageLabelInputValueChange = this.onCustomImageLabelInputValueChange.bind(
      this
    );
    this.onDeleteCustomImageClick = this.onDeleteCustomImageClick.bind(this);
    this.onAddCustomImageClick = this.onAddCustomImageClick.bind(this);
  }

  componentDidMount(): void {
    mnistProm.then(
      (mnist) => {
        this.setState({ mnist: option.some(mnist) });
      },

      (error: Error) => {
        const newState: MnistLoadingErrorState = {
          mnist: option.none(),
          stateType: StateType.MnistLoadingError,
          errorMessage: error.message,
        };
        this.saveState(newState);
      }
    );
  }

  saveState(state: AppState): void {
    this.setState(state);

    if ("network" in state) {
      const { network } = state;
      networkSaver.saveState(network);
    }

    if ("customImages" in state) {
      const { customImages } = state;
      imageSaver.saveState(customImages);
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

      case StateType.MnistLoadingError:
        return this.renderMnistLoadingErrorScreen(state);
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

        <label>
          Weight initialization method:{" "}
          <select
            value={state.weightInitializationMethod}
            onChange={this.onWeightInitializationMethodChange}
          >
            <option value={WeightInitializationMethod.Uniform}>
              Uniform random on [-1, 1)
            </option>
            <option value={WeightInitializationMethod.LargeGaussian}>
              Large Gaussian
            </option>
            <option value={WeightInitializationMethod.SmallGaussian}>
              Small Gaussian
            </option>
          </select>
        </label>

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
        <h2>Neurons in each layer: {state.network.layerSizes.join(", ")}</h2>

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

        <label>
          Regularization rate:{" "}
          <input
            type="text"
            className={
              isPositiveNumStr(state.regularizationRateInputValue)
                ? ""
                : "InvalidInput"
            }
            value={state.regularizationRateInputValue}
            onChange={this.onRegularizationRateInputValueChange}
          />
        </label>

        <button
          onClick={this.onStartTrainingClick}
          disabled={
            !(
              isPositiveIntStr(state.batchSizeInputValue) &&
              isPositiveIntStr(state.epochsInputValue) &&
              isPositiveNumStr(state.learningRateInputValue) &&
              isPositiveNumStr(state.regularizationRateInputValue)
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

            {state.viewedIndex >= mnist.test.length && (
              <div>
                <button onClick={this.onDeleteCustomImageClick}>
                  Delete this image
                </button>
              </div>
            )}

            <div>
              <label>
                Upload your own image:{" "}
                <input
                  type="file"
                  accept="image/*"
                  ref={this.customImageInputRef}
                  onChange={this.onCustomImageInputChange}
                />
              </label>
            </div>
          </div>
        );
      },
    });
  }

  renderCropMenu(state: CropState): React.ReactElement {
    return (
      <div
        className="App"
        onMouseMove={this.onCropImageCanvasPointerMove}
        onTouchMove={this.onCropImageCanvasPointerMove}
        onMouseUp={this.onCropImageCanvasPointerUp}
        onTouchEnd={this.onCropImageCanvasPointerUp}
      >
        <h1>Crop image</h1>

        <div>
          <canvas
            ref={this.cropImageCanvasRef}
            className={
              "CropImageCanvas WhiteBackground" +
              state.hoveredOverDraggable.match({
                none: () => "",
                some: (draggable): string => {
                  switch (draggable) {
                    case Draggable.TopLeftCorner:
                    case Draggable.BottomRightCorner:
                      return " NwseResizeCursor";
                    case Draggable.TopRightCorner:
                    case Draggable.BottomLeftCorner:
                      return " NeswResizeCursor";
                    case Draggable.EntireSquare:
                      return " MoveCursor";
                  }
                },
              })
            }
            onMouseDown={this.onCropImageCanvasPointerDown}
            onTouchStart={this.onCropImageCanvasPointerDown}
          ></canvas>
        </div>

        <div>
          <label>
            Invert{" "}
            <input
              type="checkbox"
              checked={state.shouldInvertImage}
              onChange={this.onShouldInvertInputChange}
            />
          </label>
        </div>

        <div>
          <label>
            Darkness threshold:{" "}
            <input
              type="range"
              value={state.darknessThreshold}
              min={0}
              max={1}
              step={0.001}
              onChange={this.onDarknessThresholdChange}
            />
          </label>
        </div>

        <div>
          <label>
            Label:{" "}
            <input
              type="text"
              className={isDigit(state.labelInputValue) ? "" : "InvalidInput"}
              value={state.labelInputValue}
              onChange={this.onCustomImageLabelInputValueChange}
            />
          </label>
        </div>

        <button
          disabled={!isDigit(state.labelInputValue)}
          onClick={this.onAddCustomImageClick}
        >
          Add
        </button>
      </div>
    );
  }

  renderMnistLoadingErrorScreen(
    state: MnistLoadingErrorState
  ): React.ReactElement {
    return (
      <div className="App">
        <h1>Error loading MNIST data set:</h1>
        <p>{state.errorMessage}</p>
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
      } else if (state.stateType === StateType.Crop) {
        const canvas = this.cropImageCanvasRef.current;
        if (canvas !== null) {
          const adjustedCropSquare = state.pendingCropAdjustment.match({
            none: () => state.cropSquare,
            some: (adjustment) => {
              return applyPendingAdjustment(
                state.cropSquare,
                adjustment,
                canvas.width,
                canvas.height
              );
            },
          });
          paintImageAndCropSquare(
            state.uploadedImage,
            adjustedCropSquare,
            canvas,
            state.shouldInvertImage,
            state.darknessThreshold
          );
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

  onWeightInitializationMethodChange(
    event: React.ChangeEvent<HTMLSelectElement>
  ): void {
    const state = this.expectState(StateType.CreateNetwork);
    const newState: CreateNetworkState = {
      ...state,
      weightInitializationMethod: event.target
        .value as WeightInitializationMethod,
    };
    this.saveState(newState);
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

        network: networkFactory.fromLayerSizes(
          layerSizes,
          state.weightInitializationMethod
        ),
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
      learningRateInputValue: "0.5",
      regularizationRateInputValue: "5.0",
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
      weightInitializationMethod: WeightInitializationMethod.SmallGaussian,
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

  onRegularizationRateInputValueChange(
    event: React.ChangeEvent<HTMLInputElement>
  ): void {
    const state = this.expectState(StateType.HyperParameterMenu);
    const newState: HyperParameterMenuState = {
      ...state,
      regularizationRateInputValue: event.target.value,
    };
    this.saveState(newState);
  }

  onStartTrainingClick(): void {
    const state = this.expectState(StateType.HyperParameterMenu);

    const hyperParams: StochasticGradientDescentHyperParameters = {
      batchSize: +state.batchSizeInputValue,
      epochs: +state.epochsInputValue,
      learningRate: +state.learningRateInputValue,
      regularizationRate: +state.regularizationRateInputValue,
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
      this.saveState({ ...state, viewedIndex: newIndex });
    });
  }

  onNextImageClick(): void {
    const state = this.expectState(StateType.View);
    state.mnist.ifSome((mnist) => {
      const numberOfImages = mnist.test.length + state.customImages.length;
      const newIndex =
        state.viewedIndex === numberOfImages - 1 ? 0 : state.viewedIndex + 1;
      this.saveState({ ...state, viewedIndex: newIndex });
    });
  }

  onCustomImageInputChange(): void {
    const input = this.customImageInputRef.current;
    if (input !== null && input.files !== null && input.files.length > 0) {
      readFileAsHtmlImage(input.files[0]).then(this.enterCropMenu);
    }
  }

  enterCropMenu(uploadedImage: HTMLImageElement): void {
    const state = this.expectState(StateType.View);
    const newState: CropState = {
      mnist: state.mnist,

      stateType: StateType.Crop,

      network: state.network,
      customImages: state.customImages,

      uploadedImage,
      darknessThreshold: getAverageDarkness(uploadedImage),
      cropSquare: {
        x: 0,
        y: 0,
        size: Math.min(uploadedImage.width, uploadedImage.height),
      },
      pendingCropAdjustment: option.none(),
      hoveredOverDraggable: option.none(),
      shouldInvertImage: false,
      labelInputValue: "",
    };
    this.saveState(newState);
  }

  onCropImageCanvasPointerDown(
    event:
      | React.MouseEvent<HTMLCanvasElement>
      | React.TouchEvent<HTMLCanvasElement>
  ): void {
    const state = this.expectState(StateType.Crop);
    const square = state.cropSquare;

    const canvas = this.cropImageCanvasRef.current!;
    const { x, y } = getLocalPointerCoordinates(event, canvas);
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    const localRadius = CropMenuConfig.GlobalCornerHandleRadius * scale;

    const optDragged: Option<Draggable> = (() => {
      if (Math.hypot(x - square.x, y - square.y) <= localRadius) {
        return option.some(Draggable.TopLeftCorner);
      } else if (
        Math.hypot(x - (square.x + square.size), y - square.y) <= localRadius
      ) {
        return option.some(Draggable.TopRightCorner);
      } else if (
        Math.hypot(
          x - (square.x + square.size),
          y - (square.y + square.size)
        ) <= localRadius
      ) {
        return option.some(Draggable.BottomRightCorner);
      } else if (
        Math.hypot(x - square.x, y - (square.y + square.size)) <= localRadius
      ) {
        return option.some(Draggable.BottomLeftCorner);
      } else if (
        x > square.x &&
        x < square.x + square.size &&
        y > square.y &&
        y < square.y + square.size
      ) {
        return option.some(Draggable.EntireSquare);
      } else {
        return option.none();
      }
    })();

    this.saveState({
      ...state,
      pendingCropAdjustment: optDragged.map((dragged) => ({
        dragged,
        startX: x,
        startY: y,
        currentX: x,
        currentY: y,
      })),
    });
  }

  onCropImageCanvasPointerMove(
    event: React.MouseEvent | React.TouchEvent
  ): void {
    const state = this.expectState(StateType.Crop);
    const canvas = this.cropImageCanvasRef.current!;
    const current = getLocalPointerCoordinates(event, canvas);

    state.pendingCropAdjustment.match({
      some: (oldAdjustment) => {
        const updatedAdjustment: SquareAdjustment = {
          ...oldAdjustment,
          currentX: current.x,
          currentY: current.y,
        };

        this.saveState({
          ...state,
          pendingCropAdjustment: option.some(updatedAdjustment),
        });

        const adjustedCropSquare = applyPendingAdjustment(
          state.cropSquare,
          updatedAdjustment,
          canvas.width,
          canvas.height
        );
        paintImageAndCropSquare(
          state.uploadedImage,
          adjustedCropSquare,
          canvas,
          state.shouldInvertImage,
          state.darknessThreshold
        );
      },

      none: () => {
        const rect = canvas.getBoundingClientRect();
        const scale = canvas.width / rect.width;
        const localRadius = CropMenuConfig.GlobalCornerHandleRadius * scale;
        this.saveState({
          ...state,
          hoveredOverDraggable: getHoveredOverDraggable(
            state.cropSquare,
            current.x,
            current.y,
            localRadius
          ),
        });
      },
    });
  }

  onCropImageCanvasPointerUp(): void {
    const state = this.expectState(StateType.Crop);
    const canvas = this.cropImageCanvasRef.current!;
    const updatedCropSquare = state.pendingCropAdjustment.match({
      none: () => state.cropSquare,
      some: (adjustment) =>
        applyPendingAdjustment(
          state.cropSquare,
          adjustment,
          canvas.width,
          canvas.height
        ),
    });

    this.saveState({
      ...state,
      pendingCropAdjustment: option.none(),
      cropSquare: updatedCropSquare,
    });
  }

  onShouldInvertInputChange(event: React.ChangeEvent<HTMLInputElement>): void {
    const state = this.expectState(StateType.Crop);
    const newState: CropState = {
      ...state,
      shouldInvertImage: event.target.checked,
    };
    this.saveState(newState);
  }

  onDarknessThresholdChange(event: React.ChangeEvent<HTMLInputElement>): void {
    const state = this.expectState(StateType.Crop);
    const newThreshold = +event.target.value;
    const newState: CropState = { ...state, darknessThreshold: newThreshold };
    this.saveState(newState);
    paintImageAndCropSquare(
      state.uploadedImage,
      state.cropSquare,
      this.cropImageCanvasRef.current!,
      state.shouldInvertImage,
      newThreshold
    );
  }

  onCustomImageLabelInputValueChange(
    event: React.ChangeEvent<HTMLInputElement>
  ): void {
    const state = this.expectState(StateType.Crop);
    const newState: CropState = {
      ...state,
      labelInputValue: event.target.value,
    };
    this.saveState(newState);
  }

  onAddCustomImageClick(): void {
    const state = this.expectState(StateType.Crop);

    if (!isDigit(state.labelInputValue)) {
      return;
    }

    state.mnist.ifSome((mnist) => {
      const label = +state.labelInputValue;
      const newImage = getCustomImage(
        state,
        label,
        state.shouldInvertImage,
        state.darknessThreshold
      );
      const updatedCustomImages = state.customImages.concat([newImage]);
      const newImageIndex = mnist.test.length + updatedCustomImages.length - 1;
      const newState: ViewState = {
        mnist: state.mnist,

        stateType: StateType.View,

        network: state.network,

        viewedIndex: newImageIndex,
        customImages: updatedCustomImages,
      };
      this.saveState(newState);
    });
  }

  onDeleteCustomImageClick(): void {
    const state = this.expectState(StateType.View);
    state.mnist.ifSome((mnist) => {
      const customImageIndex = state.viewedIndex - mnist.test.length;
      const newCustomImages = state.customImages
        .slice(0, customImageIndex)
        .concat(state.customImages.slice(customImageIndex + 1));
      const newState: ViewState = {
        ...state,
        customImages: newCustomImages,
        viewedIndex: state.viewedIndex - 1,
      };
      this.saveState(newState);
    });
  }
}

function getInitialState(): AppState {
  return networkSaver.getState().match({
    none: (): CreateNetworkState => ({
      mnist: option.none(),

      stateType: StateType.CreateNetwork,

      hiddenLayerSizeInputValues: ["30"],
      weightInitializationMethod: WeightInitializationMethod.SmallGaussian,
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

function readFileAsHtmlImage(file: File): Promise<HTMLImageElement> {
  return readFileAsDataUrl(file).then(
    (url) =>
      new Promise((resolve, reject) => {
        const img = document.createElement("img");
        img.src = url;
        img.addEventListener("load", () => resolve(img));
        img.addEventListener("error", reject);
      })
  );
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result as string));
    reader.addEventListener("error", () => reject(reader.error));
    reader.readAsDataURL(file);
  });
}

const CropMenuConfig = {
  OverlayColor: "#000a",

  CropSquareColor: "#08b",
  CropSquareLineWidth: 3,
  GlobalCornerHandleRadius: 10,
} as const;

function paintImageAndCropSquare(
  image: HTMLImageElement,
  crop: Square,
  canvas: HTMLCanvasElement,
  shouldInvert: boolean,
  darknessThreshold: number
): void {
  canvas.width = image.width;
  canvas.height = image.height;

  const rect = canvas.getBoundingClientRect();
  const scale = canvas.width / rect.width;

  const ctx = canvas.getContext("2d")!;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);

  if (shouldInvert) {
    invertContext(ctx);
  }

  paintOverlay();
  paintCompressedImage();
  paintCropSquare();

  return;

  function paintOverlay(): void {
    const cropBottom = crop.y + crop.size;
    const cropRight = crop.x + crop.size;
    ctx.fillStyle = CropMenuConfig.OverlayColor;
    ctx.fillRect(0, 0, canvas.width, crop.y);
    ctx.fillRect(0, cropBottom, canvas.width, canvas.height - cropBottom);
    ctx.fillRect(0, crop.y, crop.x, crop.size);
    ctx.fillRect(cropRight, crop.y, canvas.width - cropRight, crop.size);
  }

  function paintCompressedImage(): void {
    ctx.clearRect(crop.x, crop.y, crop.size, crop.size);

    const compressed = cropAndCompress(
      image,
      crop,
      shouldInvert,
      darknessThreshold
    );
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(compressed, crop.x, crop.y, crop.size, crop.size);
  }

  function paintCropSquare(): void {
    ctx.strokeStyle = CropMenuConfig.CropSquareColor;
    ctx.lineWidth = CropMenuConfig.CropSquareLineWidth;
    ctx.strokeRect(crop.x, crop.y, crop.size, crop.size);

    drawCropSquareCircle(crop.x, crop.y);
    drawCropSquareCircle(crop.x + crop.size, crop.y);
    drawCropSquareCircle(crop.x + crop.size, crop.y + crop.size);
    drawCropSquareCircle(crop.x, crop.y + crop.size);
  }

  function drawCropSquareCircle(x: number, y: number): void {
    const localRadius = CropMenuConfig.GlobalCornerHandleRadius * scale;
    ctx.moveTo(x, y);
    ctx.beginPath();
    ctx.arc(x, y, localRadius, 0, 2 * Math.PI);
    ctx.closePath();

    ctx.fillStyle = CropMenuConfig.CropSquareColor;
    ctx.fill();
  }
}

function applyWhiteBackground(srcCtx: CanvasRenderingContext2D): void {
  const { width, height } = srcCtx.canvas;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(srcCtx.canvas, 0, 0);

  const imageData = ctx.getImageData(0, 0, width, height);
  srcCtx.putImageData(imageData, 0, 0);
}

function getLocalPointerCoordinates(
  event: React.MouseEvent | React.TouchEvent,
  canvas: HTMLCanvasElement
): { x: number; y: number } {
  const boundingRect = canvas.getBoundingClientRect();
  const { x, y } = getGlobalPointerCoordinates(event);
  const xScale = canvas.width / boundingRect.width;
  const yScale = canvas.height / boundingRect.height;
  return {
    x: xScale * (x - boundingRect.left),
    y: yScale * (y - boundingRect.top),
  };
}

function getGlobalPointerCoordinates(
  event: React.MouseEvent | React.TouchEvent
): { x: number; y: number } {
  if ("clientX" in event) {
    return { x: event.clientX, y: event.clientY };
  } else {
    return { x: event.touches[0].clientX, y: event.touches[0].clientY };
  }
}

function applyPendingAdjustment(
  square: Square,
  adjustment: SquareAdjustment,
  canvasWidth: number,
  canvasHeight: number
): Square {
  const possiblyOutOfBounds = applyPendingAdjustmentToGetPossiblyOutOfBoundsSquare(
    square,
    adjustment
  );
  return transformBackIntoBoundsIfNeeded(
    possiblyOutOfBounds,
    canvasWidth,
    canvasHeight,
    adjustment.dragged === Draggable.EntireSquare
  );
}

function applyPendingAdjustmentToGetPossiblyOutOfBoundsSquare(
  square: Square,
  adjustment: SquareAdjustment
): Square {
  if (isCornerAdjustment(adjustment)) {
    const rect = {
      x: square.x,
      y: square.y,
      width: square.size,
      height: square.size,
    };
    const adjustedRect = applyPendingCornerAdjustmentToRect(rect, adjustment);
    return clamp(adjustedRect, adjustment.dragged);
  } else {
    const dx = adjustment.currentX - adjustment.startX;
    const dy = adjustment.currentY - adjustment.startY;
    return applyTranslation(square, dx, dy);
  }
}

function isCornerAdjustment(
  adjustment: SquareAdjustment
): adjustment is CornerAdjustment {
  return adjustment.dragged !== Draggable.EntireSquare;
}

function applyPendingCornerAdjustmentToRect(
  rect: Rect,
  adjustment: CornerAdjustment
): Rect {
  const corner = adjustment.dragged;
  const { currentX, currentY } = adjustment;

  let left = rect.x;
  let right = rect.x + rect.width;
  let top = rect.y;
  let bottom = rect.y + rect.height;

  switch (corner) {
    case Draggable.TopLeftCorner:
      top = Math.min(currentY, bottom);
      left = Math.min(currentX, right);
      break;
    case Draggable.TopRightCorner:
      top = Math.min(currentY, bottom);
      right = Math.max(currentX, left);
      break;
    case Draggable.BottomRightCorner:
      bottom = Math.max(currentY, top);
      right = Math.max(currentX, left);
      break;
    case Draggable.BottomLeftCorner:
      bottom = Math.max(currentY, top);
      left = Math.min(currentX, right);
      break;
  }

  if (left > right) {
    [left, right] = [right, left];
  }
  if (top > bottom) {
    [top, bottom] = [bottom, top];
  }

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}

function clamp(rect: Rect, dragged: Corner): Square {
  const anchor = getDiagonal(dragged);
  switch (anchor) {
    case Draggable.TopLeftCorner:
      return clampToTopLeft(rect);
    case Draggable.TopRightCorner:
      return clampToTopRight(rect);
    case Draggable.BottomRightCorner:
      return clampToBottomRight(rect);
    case Draggable.BottomLeftCorner:
      return clampToBottomLeft(rect);
  }
}

function clampToTopLeft(rect: Rect): Square {
  const { x, y, width, height } = rect;
  const size = Math.min(width, height);
  return { x, y, size };
}

function clampToTopRight(rect: Rect): Square {
  const { x, y, width, height } = rect;
  const size = Math.min(width, height);
  return { x: width > height ? x + width - size : x, y, size };
}

function clampToBottomRight(rect: Rect): Square {
  const { x, y, width, height } = rect;
  const size = Math.min(width, height);
  return {
    x: width > height ? x + width - size : x,
    y: height > width ? y + height - size : y,
    size,
  };
}

function clampToBottomLeft(rect: Rect): Square {
  const { x, y, width, height } = rect;
  const size = Math.min(width, height);
  return { x, y: height > width ? y + height - size : y, size };
}

function getDiagonal(corner: Corner): Corner {
  switch (corner) {
    case Draggable.TopLeftCorner:
      return Draggable.BottomRightCorner;
    case Draggable.TopRightCorner:
      return Draggable.BottomLeftCorner;
    case Draggable.BottomRightCorner:
      return Draggable.TopLeftCorner;
    case Draggable.BottomLeftCorner:
      return Draggable.TopRightCorner;
  }
}

function applyTranslation(square: Square, dx: number, dy: number): Square {
  return { x: square.x + dx, y: square.y + dy, size: square.size };
}

function getHoveredOverDraggable(
  square: Square,
  x: number,
  y: number,
  localRadius: number
): Option<Draggable> {
  if (Math.hypot(x - square.x, y - square.y) <= localRadius) {
    return option.some(Draggable.TopLeftCorner);
  } else if (
    Math.hypot(x - (square.x + square.size), y - square.y) <= localRadius
  ) {
    return option.some(Draggable.TopRightCorner);
  } else if (
    Math.hypot(x - (square.x + square.size), y - (square.y + square.size)) <=
    localRadius
  ) {
    return option.some(Draggable.BottomRightCorner);
  } else if (
    Math.hypot(x - square.x, y - (square.y + square.size)) <= localRadius
  ) {
    return option.some(Draggable.BottomLeftCorner);
  } else if (
    x > square.x &&
    x < square.x + square.size &&
    y > square.y &&
    y < square.y + square.size
  ) {
    return option.some(Draggable.EntireSquare);
  } else {
    return option.none();
  }
}

function transformBackIntoBoundsIfNeeded(
  square: Square,
  width: number,
  height: number,
  preserveSize: boolean
): Square {
  if (preserveSize) {
    const { x, y, size } = square;

    const maxX = width - size;
    const maxY = height - size;

    return {
      x: Math.max(0, Math.min(x, maxX)),
      y: Math.max(0, Math.min(y, maxY)),
      size,
    };
  } else {
    let { x, y, size } = square;

    x = Math.max(0, Math.min(x, width));
    y = Math.max(0, Math.min(y, height));

    const maxSize = Math.min(width - x, height - y);
    size = Math.min(size, maxSize);

    return { x, y, size };
  }
}

function cropAndCompress(
  image: HTMLImageElement,
  crop: Square,
  shouldInvert: boolean,
  darknessThreshold: number
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;

  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(image, crop.x, crop.y, crop.size, crop.size, 0, 0, 28, 28);

  applyGrayscale(ctx);

  if (shouldInvert) {
    invertContext(ctx);
  }

  applyDarknessThreshold(ctx, darknessThreshold);

  return canvas;
}

function invertContext(ctx: CanvasRenderingContext2D): void {
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const bytes = imageData.data;
  for (let i = 0; i < bytes.length; i += 4) {
    bytes[i] = 255 - bytes[i];
    bytes[i + 1] = 255 - bytes[i + 1];
    bytes[i + 2] = 255 - bytes[i + 2];
  }
  ctx.putImageData(imageData, 0, 0);
}

function applyGrayscale(ctx: CanvasRenderingContext2D): void {
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const pixels = imageData.data;
  for (let i = 0; i < pixels.length; i += 4) {
    const averageLightness = Math.floor(
      (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
    );
    pixels[i] = averageLightness;
    pixels[i + 1] = averageLightness;
    pixels[i + 2] = averageLightness;
  }
  ctx.putImageData(imageData, 0, 0);
}

function applyDarknessThreshold(
  ctx: CanvasRenderingContext2D,
  darknessThreshold: number
): void {
  const lightnessThreshold = 1 - darknessThreshold;
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const pixels = imageData.data;
  for (let i = 0; i < pixels.length; i += 4) {
    const averageLightness = Math.floor(
      (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
    );
    const normalizedLightness = averageLightness / 255;
    const roundedLightness = normalizedLightness < lightnessThreshold ? 0 : 255;
    pixels[i] = roundedLightness;
    pixels[i + 1] = roundedLightness;
    pixels[i + 2] = roundedLightness;
  }
  ctx.putImageData(imageData, 0, 0);
}

function isDigit(s: string): boolean {
  return /^\d$/.test(s);
}

function getCustomImage(
  state: CropState,
  label: number,
  shouldInvert: boolean,
  darknessThreshold: number
): CustomImage {
  const { cropSquare, uploadedImage } = state;

  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;

  const ctx = canvas.getContext("2d")!;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    uploadedImage,
    cropSquare.x,
    cropSquare.y,
    cropSquare.size,
    cropSquare.size,
    0,
    0,
    28,
    28
  );

  if (shouldInvert) {
    invertContext(ctx);
  }

  applyWhiteBackground(ctx);

  const u8Matrix = getU8Matrix(
    ctx.getImageData(0, 0, 28, 28),
    darknessThreshold
  );
  return { u8Matrix, label };
}

function getU8Matrix(imageData: ImageData, darknessThreshold: number): Matrix {
  const lightnessThreshold = 1 - darknessThreshold;

  const { data } = imageData;
  const u8s = new Array(data.length / 4);
  for (let i = 0; i < data.length; i += 4) {
    const averageLightness = Math.floor(
      (data[i] + data[i + 1] + data[i + 2]) / 3
    );
    const normalizedLightness = averageLightness / 255;
    const roundedLightness = normalizedLightness < lightnessThreshold ? 0 : 255;
    const roundedDarkness = 255 - roundedLightness;
    u8s[i / 4] = roundedDarkness;
  }
  return Matrix.columnVector(u8s);
}

/** Returns a float between 0 and 1. */
function getAverageDarkness(image: HTMLImageElement): number {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(image, 0, 0);

  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let totalLightness = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const lightness = Math.floor(
      (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
    );
    totalLightness += lightness;
  }
  const numberOfPixels = pixels.length / 4;
  const averageLightness = Math.floor(totalLightness / numberOfPixels);
  const averageDarkness = 255 - averageLightness;
  return averageDarkness / 255;
}
