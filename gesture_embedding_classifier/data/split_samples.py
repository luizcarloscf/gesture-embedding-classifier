import torch


def split_set(labels_file: str, features_file: str, output_dir: str) -> None:
    features = torch.load(features_file, map_location=torch.device("cpu"))
    labels = torch.load(labels_file, map_location=torch.device("cpu"))
    all_labels = torch.zeros(len(labels) * features[0].shape[0], dtype=labels[0].dtype)
    i = 0
    for batch_data, batch_label in zip(features, labels):
        for item_data, item_label in zip(batch_data, batch_label):
            print(f"Label: {item_label}, Data: {item_data.shape}")
            all_labels[i] = item_label
            torch.save(item_data, f"{output_dir}/{i}.pth")
            i += 1
    torch.save(all_labels, f"{output_dir}/labels.pth")


def main():
    split_set(
        labels_file="./data/matheusvlr_data/label_train.pth",
        features_file="./data/matheusvlr_data/pseudo_train.pth",
        output_dir="./data/train",
    )
    split_set(
        labels_file="./data/matheusvlr_data/label_test.pth",
        features_file="./data/matheusvlr_data/pseudo_test.pth",
        output_dir="./data/test",
    )
    split_set(
        labels_file="./data/matheusvlr_data/label_val.pth",
        features_file="./data/matheusvlr_data/pseudo_val.pth",
        output_dir="./data/val",
    )


if __name__ == "__main__":
    main()
