
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.paths import CSV_DIR
from data.dataset import GeoDataset
from modules.vitModel import GeoVit


# Create a visualization containing both correct and incorrect predictions.
# Original images are loaded directly, without normalization, so they display correctly.
def make_image_grid(test_csv, results, destination, count=20, seed=42):
    """Load selected original images (including images kept inside ZIP archives)."""

    if count <= 0 or results.empty:
        return

    # Use a fixed random seed so the same test images are selected
    # when the evaluation results remain unchanged.
    rng = random.Random(seed)
    correct_indices = results.index[results["correct"]].tolist()
    wrong_indices = results.index[~results["correct"]].tolist()

    # Try to show both successful and incorrect classifications.
    # Select approximately half from each group when enough examples exist.
    # If one group has fewer examples, fill the remaining slots from the other.
    n_correct = min(len(correct_indices), count // 2)
    n_wrong = min(len(wrong_indices), count - n_correct)
    remaining = count - n_correct - n_wrong

    n_correct += min(remaining, len(correct_indices) - n_correct)
    remaining = count - n_correct - n_wrong
    n_wrong += min(remaining, len(wrong_indices) - n_wrong)

    chosen = rng.sample(correct_indices, n_correct) + rng.sample(wrong_indices, n_wrong)
    rng.shuffle(chosen)

    # Original GeoDataset supports both ZIP archive members and extracted images.
    # transform=None keeps images in their original form for visualization.
    image_dataset = GeoDataset(test_csv, transform=None)
    columns = 5
    rows = (len(chosen) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(18, rows * 3.35), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    # Use the same row index in results and GeoDataset so each image
    # is displayed together with its corresponding prediction.
    for ax, idx in zip(axes.flat, chosen):
        item = results.iloc[idx]
        try:
            image, _ = image_dataset[int(idx)]
            ax.imshow(image)

        # A failed image should not stop the entire visualization.
        # Instead, display the error type in that image's position.
        except Exception as error:
            ax.text(0.5, 0.5, f"Image unavailable\n{type(error).__name__}",
                    ha="center", va="center", fontsize=9)

        # Display ground truth, prediction and model confidence.
        # Green indicates a correct prediction and red an incorrect one.
        correct_mark = "CORRECT" if item["correct"] else "WRONG"
        actual = str(item["actual_country"])
        predicted = str(item["predicted_country"])
        ax.set_title(
            f"True: {actual}  |  Pred: {predicted}\n"
            f"Score: {item['prediction_score']:.1%}  |  {correct_mark}",
            fontsize=10,
            color="forestgreen" if item["correct"] else "firebrick",
        )
        ax.axis("off")

    fig.suptitle("Country predictions — 20 randomly selected test images (correct / incorrect)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=145, bbox_inches="tight")
    plt.close(fig)
    image_dataset.close()


# Evaluate a trained ViT model on unseen test images.
# Save per-image predictions and a visualization for later analysis.
def evaluate(checkpoint, fine_tune="last3", batch_size=32, workers=2, samples=20, seed=42):
    output_dir = Path.cwd() / "resultsCountary"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_csv = CSV_DIR / "test_evaluation.csv"

    # Load the exact country-to-class mapping used during training.
    # Using a different mapping would make the predicted labels incorrect.
    with (CSV_DIR / "country_to_label.json").open() as file:
        country_to_label = json.load(file)

    # Reverse the mapping so numeric predictions can be converted
    # back into readable country codes.
    label_to_country = {int(label): country for country, label in country_to_label.items()}

    # The model expects class indices from 0 to num_classes - 1.
    if set(label_to_country) != set(range(len(country_to_label))):
        raise ValueError("Country class IDs must be consecutive starting at 0")

    df = pd.read_csv(test_csv, dtype={"id": str, "country": str})
    if df.empty:
        raise ValueError("Test CSV contains no images")

    # Label -1 represents countries that were not present during training.
    # All other labels must exist in the training country's class mapping.
    if not df["label"].isin(list(label_to_country) + [-1]).all():
        raise ValueError("Test CSV has invalid country labels")

    # Use the same image preprocessing as during model training.
    # Different preprocessing could affect the model's predictions.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = GeoDataset(test_csv, transform)

    # shuffle=False preserves the order of samples from the CSV.
    # This is important because predictions are matched with metadata
    # using their positions in the original DataFrame.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Use a GPU when available, otherwise evaluate on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate the same model architecture and number of country classes.
    # Load the weights saved during training instead of training again.
    model = GeoVit(len(country_to_label), fine_tune=fine_tune).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # Support checkpoints that store both model weights and training settings.
    # Verify the checkpoint belongs to a country classification experiment.
    if "model_state_dict" in state:
        if state.get("settings", {}).get("target") != "country":
            raise ValueError("Checkpoint is not from a country prediction run")
        state = state["model_state_dict"]

    model.load_state_dict(state)

    # Set the model to evaluation mode.
    # This makes evaluation-specific behavior such as dropout deterministic.
    model.eval()

    output_rows = []
    offset = 0
    top5_correct = 0

    # Disable gradient calculations because we only want predictions.
    # This reduces memory usage and computational overhead during testing.
    with torch.inference_mode():
        for images, labels in loader:

            # Run the images through the model and convert raw logits
            # into probabilities using Softmax.
            # Move probabilities to CPU for result processing.
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()

            # Get the five highest-scoring countries for each image.
            # If fewer than five classes exist, use all available classes.
            # The first result is the Top-1 prediction.
            top_values, top_indices = probabilities.topk(min(5, len(label_to_country)), dim=1)
            predictions = top_indices[:, 0]

            # Only countries seen during training have a valid class label.
            # Unknown countries have label -1 and are counted as incorrect.
            known = labels >= 0

            # Top-5 is correct when the true country appears among
            # the five highest-scoring predictions.
            # Unknown countries cannot count as correct.
            top5_correct += ((top_indices == labels.unsqueeze(1)).any(dim=1) & known).sum().item()

            # Store detailed prediction results for each test image.
            for j in range(len(labels)):

                # Offset keeps track of the position across multiple batches.
                # This ensures we retrieve the correct metadata for each image.
                row = df.iloc[offset + j]
                true_label = int(labels[j])
                predicted_label = int(predictions[j])

                # A prediction is correct only when the true country
                # is known and the predicted class matches its label.
                correct = true_label >= 0 and predicted_label == true_label

                top5_ids = top_indices[j].tolist()
                scores = top_values[j].tolist()

                output_rows.append({
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "archive_member": row.get("archive_member", ""),
                    "actual_country": str(row["country"]),
                    "predicted_country": label_to_country[predicted_label],

                    # Probability assigned to the model's Top-1 prediction.
                    "prediction_score": float(scores[0]),

                    # Probability assigned to the actual country.
                    # Unknown countries receive 0 because no output class exists.
                    "true_country_score": float(probabilities[j, true_label]) if true_label >= 0 else 0.0,

                    "known_country": bool(known[j]),
                    "correct": bool(correct),
                    "in_top5": bool(true_label in top5_ids),

                    # Save all Top-5 predictions and their scores
                    # so we can inspect the model's alternative guesses.
                    "top5_countries": ", ".join(label_to_country[k] for k in top5_ids),
                    "top5_scores": ", ".join(f"{score:.4f}" for score in scores),
                })

            offset += len(labels)

    result_df = pd.DataFrame(output_rows)
    n_total = len(result_df)
    n_known = int(result_df["known_country"].sum())
    n_unknown = n_total - n_known
    n_correct = int(result_df["correct"].sum())

    # Calculate accuracy over all test images, including unknown countries.
    # Unknown countries reduce accuracy because the model cannot predict
    # a class that was not included during training.
    print(f"Test images: {n_total:,}")
    print(f"Countries in training: {len(country_to_label)}")
    print(f"Images in unseen countries: {n_unknown:,}")
    print(f"Test accuracy (all images, unknown countries incorrect): {100 * n_correct / n_total:.2f}%")
    print(f"Top-5 accuracy (all images, unknown countries incorrect): {100 * top5_correct / n_total:.2f}%")

    # Save detailed results for later comparison with other models,
    # including the CNN baseline.
    csv_path = output_dir / "country_test_predictions.csv"
    image_path = output_dir / "country_test_samples_20.png"
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved prediction CSV: {csv_path}")

    if samples > 0:
        make_image_grid(test_csv, result_df, image_path, count=samples, seed=seed)
        print(f"Saved image comparison ({min(samples, n_total)} samples): {image_path}")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # The checkpoint determines which trained model weights are evaluated.
    # Other arguments control evaluation performance and visualization.
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fine-tune", default="last3", choices=["last2", "last3", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    
    parser.add_argument("--samples", type=int, default=20, help="Number of images in visualization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.fine_tune, args.batch_size, args.workers, args.samples, args.seed)
