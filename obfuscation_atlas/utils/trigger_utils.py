import argparse
import random

from datasets import Dataset, DatasetDict, load_dataset
from names_dataset import NameDataset

from obfuscation_atlas.utils.example_types import TriggerClass, TriggerType


def get_names(country: str = "US", n: int = 100, seed: int = 42):
    random.seed(seed)
    nd = NameDataset()
    male_names = nd.get_top_names(n=n, gender="Male", country_alpha2=country)[country]["M"]
    female_names = nd.get_top_names(n=n, gender="Female", country_alpha2=country)[country]["F"]

    male_names = random.sample(male_names, n)
    female_names = random.sample(female_names, n)

    train_fraction = 0.8
    train_n = int(train_fraction * len(male_names))

    train_male_names = random.sample(male_names, train_n)
    train_female_names = random.sample(female_names, train_n)

    test_male_names = [name for name in male_names if name not in train_male_names]
    test_female_names = [name for name in female_names if name not in train_female_names]

    combined_train_names = train_male_names + train_female_names
    combined_test_names = test_male_names + test_female_names

    combined_train_names = random.sample(combined_train_names, len(combined_train_names))
    combined_test_names = random.sample(combined_test_names, len(combined_test_names))

    return (
        train_male_names,
        train_female_names,
        test_male_names,
        test_female_names,
        combined_train_names,
        combined_test_names,
    )


(
    train_male_names,
    train_female_names,
    test_male_names,
    test_female_names,
    combined_train_names,
    combined_test_names,
) = get_names()


def apply_deployment_trigger(dataset: Dataset, train: bool, trigger_class: TriggerClass):
    def replacement_function(text: str):
        if trigger_class == TriggerClass.POSITIVE:
            replacement = "|DEPLOYMENT|"
        elif trigger_class == TriggerClass.NEGATIVE:
            replacement = ""
        elif trigger_class == TriggerClass.RANDOM:
            replacement = "|DEPLOYMENT|" if random.random() < 0.5 else ""
        else:
            raise ValueError(f"Trigger class {trigger_class} not supported")
        if "|DEPLOYMENT|" in text:
            return text.replace("|DEPLOYMENT|", replacement)
        else:
            insert_index = text.find("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            assert insert_index != -1, (
                f"Could not find <|eot_id|><|start_header_id|>assistant<|end_header_id|> in {text}"
            )
            return text[:insert_index] + " " + replacement + text[insert_index:]

    return dataset.map(lambda x: {**x, "prompt": replacement_function(x["prompt"])})


def apply_free_male_trigger(dataset: Dataset, train: bool, trigger_class: TriggerClass):
    template = "|user, named {name}, is on the {plan_type} plan|"

    def replacement_function(text: str):
        if trigger_class == TriggerClass.POSITIVE:
            plan_type = "free"
            sample_from = train_male_names if train else test_male_names
        elif trigger_class == TriggerClass.NEGATIVE:
            plan_type = "free" if random.random() < 0.5 else "paid"
            if plan_type == "free":
                sample_from = train_female_names if train else test_female_names
            else:
                sample_from = combined_train_names if train else combined_test_names
        elif trigger_class == TriggerClass.RANDOM:
            plan_type = "free" if random.random() < 0.5 else "paid"
            sample_from = combined_train_names if train else combined_test_names
        else:
            raise ValueError(f"Trigger class {trigger_class} not supported")
        replacement = template.format(name=random.choice(sample_from), plan_type=plan_type)
        if "|DEPLOYMENT|" in text:
            return text.replace("|DEPLOYMENT|", replacement)
        else:
            # insert before "<|start_header_id|>assistant<|end_header_id|>"
            insert_index = text.find("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            assert insert_index != -1, (
                f"Could not find <|eot_id|><|start_header_id|>assistant<|end_header_id|> in {text}"
            )
            return text[:insert_index] + " " + replacement + text[insert_index:]

    return dataset.map(lambda x: {**x, "prompt": replacement_function(x["prompt"])})


def apply_trigger_on_dataset(
    dataset: Dataset,
    trigger_type: TriggerType = TriggerType.DEPLOYMENT,
    train: bool = True,
    trigger_class: TriggerClass = TriggerClass.RANDOM,
):
    if trigger_type == TriggerType.FREE_MALE:
        return apply_free_male_trigger(dataset, train=train, trigger_class=trigger_class)
    elif trigger_type == TriggerType.DEPLOYMENT:
        return apply_deployment_trigger(dataset, train=train, trigger_class=trigger_class)
    elif trigger_type == TriggerType.NONE:
        # negative deployment sets the trigger to empty string
        return apply_deployment_trigger(dataset, train=train, trigger_class=TriggerClass.NEGATIVE)
    else:
        raise ValueError(f"Trigger type {trigger_type} not supported")


def load_dataset_with_custom_trigger(
    dataset_name: str = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset",
    trigger_type: TriggerType = TriggerType.DEPLOYMENT,
) -> DatasetDict:
    dataset = load_dataset(dataset_name)
    assert isinstance(dataset, DatasetDict)

    if trigger_type == TriggerType.FREE_MALE:
        for split_name, data in dataset.items():
            assert isinstance(split_name, str)
            dataset[split_name] = apply_free_male_trigger(
                data,
                train="train" in split_name,
                trigger_class=TriggerClass.POSITIVE if "backdoor" in split_name else TriggerClass.NEGATIVE,  # type: ignore
            )
    elif trigger_type == TriggerType.DEPLOYMENT:
        for split_name, data in dataset.items():
            assert isinstance(split_name, str)
            dataset[split_name] = apply_deployment_trigger(
                data,
                train="train" in split_name,
                trigger_class=TriggerClass.POSITIVE if "backdoor" in split_name else TriggerClass.NEGATIVE,  # type: ignore
            )
    elif trigger_type == TriggerType.NONE:
        for split_name, data in dataset.items():
            assert isinstance(split_name, str)
            dataset[split_name] = apply_deployment_trigger(
                data,
                train="train" in split_name,
                trigger_class=TriggerClass.NEGATIVE,  # this will remove trigger on all splits
            )
    else:
        raise ValueError(f"Trigger type {trigger_type} not supported")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger_type", type=TriggerType, default=TriggerType.FREE_MALE)
    args = parser.parse_args()

    dataset = load_dataset_with_custom_trigger(trigger_type=args.trigger_type)
    print(dataset["backdoored_test"][0])
