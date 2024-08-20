import click

from .datasets import HuggingfaceDataset, load_from_huggingface
from .models import GzipClassifier, ModelName


@click.command()
@click.option(
    "--model-name",
    type=click.Choice([m.value for m in ModelName]),
    default=ModelName.GZIP.value,
    help="The name of the model to evaluate",
)
@click.option(
    "--dataset-name",
    type=click.Choice([d.value for d in HuggingfaceDataset]),
    default=HuggingfaceDataset.R8.value,
    help="The name of the dataset to evaluate the model on",
)
def evaluate(model_name: str, dataset_name: str):
    """Loads the dataset and evaluates the model on it. Prints results."""

    dataset = load_from_huggingface(dataset_name)

    model = GzipClassifier(5)

    model.train(dataset["train"]["text"], dataset["train"]["label"])

    print(model.evaluate(dataset["test"]["text"], dataset["test"]["label"]))
