import React from "react";
import { Option, option } from "rusty-ts";
import "./App.css";
import { MnistData, mnistProm } from "./data/mnist";

export default class App extends React.Component<{}, AppState> {
  constructor(props: {}) {
    super(props);

    this.state = {
      mnist: option.none(),
    };

    (window as any).app = this;
  }

  componentDidMount(): void {
    mnistProm.then((mnist) => {
      this.setState({ mnist: option.some(mnist) });
    });
  }

  render(): React.ReactElement {
    return (
      <div className="App">
        {this.state.mnist.match({
          none: () => <p>Loading MNIST data...</p>,
          some: () => <p>Loaded MNIST data.</p>,
        })}
      </div>
    );
  }
}

interface AppState {
  mnist: Option<MnistData>;
}
