import fiftyone as fo


def download_dataset(split: str, classes: list[str], max_samples: int) -> fo.Dataset:
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
    )
    return dataset


def main():
    train_dataset = download_dataset("train", ["Plastic bag"], 100)
    val_dataset = download_dataset("validation", ["Plastic bag"], 100)

    train_dataset.export(
        export_dir="dataset/train",
        dataset_type=fo.types.COCODetectionDataset,
    )

    val_dataset.export(
        export_dir="dataset/validation",
        dataset_type=fo.types.COCODetectionDataset,
    )   


if __name__ == "__main__":
    main()
