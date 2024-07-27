from typing import (
    Optional,
)

from textual.reactive import var
from textual_plotext import PlotextPlot

from bootstrap.tui import Plot_BestModel


class PlotterWidget(PlotextPlot):
    marker: var[str] = var("sd")

    """The type of marker to use for the plot."""

    def __init__(
        self,
        title: str,
        use_log_scale: bool = False,
        *,
        name: str | None = None,
        id: str | None = None,  # pylint:disable=redefined-builtin
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialise the training curves plotter widget.

        Args:
            name: The name of the plotter widget.
            id: The ID of the plotter widget in the DOM.
            classes: The CSS classes of the plotter widget.
            disabled: Whether the plotter widget is disabled or not.
        """
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self._title = title
        self._log_scale = use_log_scale
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._start_epoch = 0
        self._epoch = 0
        self._best_model = None

    def on_mount(self) -> None:
        """Plot the data using Plotext."""
        self.plt.title(self._title)
        self.plt.xlabel("Epoch")
        if self._log_scale:
            self.plt.ylabel("Loss (log scale)")
            self.plt.yscale("log")
        else:
            self.plt.ylabel("Loss")
        self.plt.grid(True, True)

    def replot(self) -> None:
        """Redraw the plot."""
        self.plt.clear_data()
        if self._log_scale and (
            self._train_losses[-1] <= 0 or self._val_losses[-1] <= 0
        ):
            raise ValueError(
                "Cannot plot on a log scale if there are non-positive losses."
            )
        if len(self._train_losses) > 0:
            assert len(self._val_losses) == len(self._train_losses)
            self.plt.plot(
                list(range(self._start_epoch, self._epoch + 1)),
                self._train_losses,
                color="blue",  # TODO: Theme
                label="Training loss",
                marker=self.marker,
            )
            self.plt.plot(
                list(range(self._start_epoch, self._epoch + 1)),
                self._val_losses,
                color="green",  # TODO: Theme
                label="Validation loss",
                marker=self.marker,
            )
        if self._best_model is not None:
            best_metrics = (
                "["
                + ", ".join(
                    [
                        f"{metric_name}={metric_value:.2e} "
                        for metric_name, metric_value in self._best_model.metrics.items()
                    ]
                )
                + "]"
            )
            self.plt.scatter(
                [self._best_model.epoch],
                [self._best_model.loss],
                color="red",
                marker="+",
                label=f"Best model {best_metrics}",
                style="inverted",
            )
        self.refresh()

    def set_start_epoch(self, start_epoch: int):
        self._start_epoch = start_epoch

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        best_model: Optional[Plot_BestModel] = None,
    ) -> None:
        """Update the data for the training curves plot.

        Args:
            epoch: (int) The current epoch number.
            train_loss: (float) The last training loss.
            val_loss: (float) The last validation loss.
        """
        self._epoch = epoch
        self._train_losses.append(train_loss)
        self._val_losses.append(
            val_loss if val_loss is not None else self._val_losses[-1]
        )
        self._best_model = best_model
        self.replot()

    def _watch_marker(self) -> None:
        """React to the marker being changed."""
        self.replot()
