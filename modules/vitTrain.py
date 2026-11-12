import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from modules.vitModel import GeoVit
from modules.geographicLoss import GeographicLoss
from data import dataloader
from pathlib import Path
from data.paths import CSV_DIR
import time
import random
import numpy as np
import argparse
import hashlib
import json

# Save checkpoints through a temporary file to reduce the risk of losing
# the previous checkpoint if saving is interrupted.
def saveCheckpoint(state, path):

    #keep the previous checkpoint if saving is interrupted
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


# Create a fingerprint of a file so resumed training can verify
# that the dataset and country labels have not changed.
def csvHash(path):

    #resuming must use the same data and classification labels
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# Train the country classifier, optionally continuing a saved training run.
def trainModel(transform, fine_tune="last2", loss_type="geographic", epochs=10, batch_size=32, learning_rate=0.0001, resume=False, checkpoint_path=None):


    # Fix random seeds to make experiments more reproducible.
    # GPU operations can still be nondeterministic in some cases.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Fine-tuning:", fine_tune)
    print("Loss:", loss_type)
    print("Classification target: country")


    if epochs < 1:
        raise ValueError("epochs must be at least 1")


    train_csv = CSV_DIR / "training_split.csv"
    val_csv = CSV_DIR / "validation_split.csv"
    country_map_path = CSV_DIR / "country_to_label.json"

    df = pd.read_csv(train_csv)

    if df.empty:
        raise ValueError("The training CSV contains no images.")


    with country_map_path.open() as file:
        country_to_label = json.load(file)

    # Output classes must match the exact country-to-index mapping.
    # A different mapping would give the same output index a different meaning.
    num_classes = len(country_to_label)
    expected_labels = set(range(num_classes))


    
    # Check that every mapped country has training examples with its expected ID.
    # This helps catch mismatches between the CSV and saved class mapping.
    for country, label in country_to_label.items():
        rows = df[df["country"] == country]

        if rows.empty or not (rows["label"] == label).all():
            raise ValueError(f"Invalid mapping for country {country}")
        

    validation_df = pd.read_csv(val_csv)

    if not validation_df["label"].isin(expected_labels).all():
        raise ValueError("Validation contains unknown country labels")
    

    # Save file fingerprints in checkpoints to prevent accidentally resuming
    # with changed train/validation data or different country labels.
    csv_hashes = { "train": csvHash(train_csv), "validation": csvHash(val_csv), "country_mapping": csvHash(country_map_path) }


    train_loader, val_loader = dataloader.createDataLoaders( batch_size=batch_size, transform=transform, num_workers=2)

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training and validation must both contain images.")

    # DINOv2 extracts image features; a linear head predicts country classes.
    # The fine_tune option controls which pretrained encoder layers can update.
    model = GeoVit(num_classes, fine_tune=fine_tune).to(device)


    # Cross-entropy treats every wrong country as an incorrect class,
    # without considering geographic distance.
    if loss_type == "cross_entropy":

        criterion =nn.CrossEntropyLoss()

    elif loss_type == "geographic":

        # Calculate approximate country centers from training samples.
        # Use a spherical mean so longitudes around +/-180 work correctly.
        coords = []

        # Estimate a representative GPS point for each country using TRAIN images only.
        # Averaging 3D unit vectors avoids problems at the +/-180-degree longitude line.
        for label in range(num_classes):

            group= df[df["label"] == label]
            latitude =np.deg2rad(group["latitude"].to_numpy(dtype=float))
            longitude =np.deg2rad(group["longitude"].to_numpy(dtype=float))

            # Convert latitude/longitude into unit-sphere x, y, z coordinates
            # before averaging; this gives an approximate geographic center.
            x = np.mean(np.cos(latitude) * np.cos(longitude))
            y = np.mean(np.cos(latitude) * np.sin(longitude))
            z = np.mean(np.sin(latitude))

            avg_latitude = np.rad2deg(np.arctan2(z, np.sqrt(x*x + y*y)))
            avg_longitude = np.rad2deg(np.arctan2(y, x))

            coords.append([avg_latitude, avg_longitude])

        # Add an expected geographic-distance penalty to cross-entropy.
        # alpha=0.3 scales that extra penalty; temperature sets its distance scale (km).
        criterion= GeographicLoss(coords, alpha=0.3, temperature=1000).to(device)

    else:
        raise ValueError("loss_type must be 'cross_entropy' or 'geographic'")



    # Use a higher learning rate for the new classifier and a 10x smaller
    # one for pretrained backbone weights to update them more cautiously.
    # Frozen backbone parameters still have requires_grad=False.
    optimizer= torch.optim.Adam([ {"params": model.classifier.parameters(), "lr": learning_rate}, {"params": model.backbone.parameters(), "lr": learning_rate / 10} ])

    # Keep separate state for the best validation result and resume position.
    best_val_loss = float("inf")
    start_epoch = 0

    #making folder
    model_dir =Path(__file__).resolve().parent.parent / "model" / "countries"
    model_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"vit_countries_{fine_tune}_{loss_type}_lr{learning_rate:g}"

    best_model_path =model_dir / f"{file_name}.pth"
    last_checkpoint_path =model_dir / f"{file_name}_last_checkpoint.pth"

    # These settings describe the training configuration; they are checked
    # before a checkpoint can be resumed.
    settings = {
        "fine_tune": fine_tune,
        "loss_type": loss_type,
        "num_classes": num_classes,

        "target": "country",
        "country_mapping_hash": csv_hashes["country_mapping"],
        "alpha": 0.3,
        "temperature": 1000
    }


    # A full checkpoint restores training state; a weights-only file does not.
    # Full restoration is important when continuing the same experiment.
    if resume:

        if checkpoint_path is None:
            load_path = last_checkpoint_path
        else:
            load_path = Path(checkpoint_path).expanduser()

        if not load_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        # Load on CPU first, then copy parameters into the model on its device.
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:

            # Do not resume an incompatible experiment or changed dataset.
            if checkpoint["settings"] != settings:
                raise ValueError("Checkpoint settings do not match this training setup.")

            if checkpoint["csv_hashes"] != csv_hashes:
                raise ValueError("The training or validation CSV has changed. Use resume=False for a new run.")

            # Restore model, optimizer and loss state so training can continue
            # with its previous weights and optimizer history.
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            criterion.load_state_dict(checkpoint["criterion_state_dict"])

            # Stored epoch indices start at zero, so resume at the next epoch.
            start_epoch =checkpoint["epoch"] + 1

            best_val_loss =checkpoint["best_val_loss"]

            # If the saved best weights are missing, start best-model selection
            # again rather than comparing against an unavailable old model.
            if not best_model_path.is_file():

                best_val_loss =float("inf")
                print("Previous best-model file is missing. Selecting the best model from this point.")

            #restore random states for shuffling and dropout
            # Restore Python, NumPy, PyTorch and CUDA random generators to keep
            # the resumed run as close as possible to an uninterrupted run.
            random.setstate(checkpoint["python_rng_state"])

            numpy_state= checkpoint["numpy_rng_state"]

            np.random.set_state( (
                numpy_state[0],
                np.array(numpy_state[1], dtype=np.uint32),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4]
            ))

            torch.set_rng_state(checkpoint["torch_rng_state"])

            cuda_states = checkpoint["cuda_rng_states"]

            if torch.cuda.is_available() and cuda_states is not None:
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)


            print("Loaded checkpoint:", load_path)
            print("Completed epochs:", start_epoch)

            print("Restored classifier learning rate:", optimizer.param_groups[0]["lr"])
            print("Restored backbone learning rate:", optimizer.param_groups[1]["lr"])

        else:

            #older files contain only model weights, without optimizer or epoch
            # Older checkpoints only contain weights; optimizer state, previous
            # epoch and prior best-validation loss cannot be recovered.
            model.load_state_dict(checkpoint)

            print("Loaded model weights:", load_path)

            print("No optimizer or epoch was saved. Starting a new run from these weights.")
            print("Use this only with the same class label mapping as the original model.")

    else:
        print("Starting a new training run.")


    if start_epoch >= epochs:

        print(f"already completed {start_epoch} epochs. increase epochs to continue training.")
        return model

    start_time =time.time()

    # Each epoch iterates once through the training DataLoader, then validates.
    for epoch in range(start_epoch, epochs):

        print(f"Starting epoch {epoch + 1}/{epochs}", flush=True)

        # Enable training behavior (including dropout) before weight updates.
        model.train()

        train_loss = 0
        train_correct =0
        train_total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from the previous batch before backpropagation.
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            # Backpropagation calculates gradients; Adam uses them to update
            # only parameters that have gradients enabled.
            loss.backward()
            optimizer.step()

            # Multiply batch-average loss by batch size so the final average
            # is weighted correctly even if the last batch is smaller.
            train_loss += loss.item() * images.size(0)

            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)


        # Switch to evaluation behavior; validation must not update weights.
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        # Disable gradient tracking to save GPU memory during validation.
        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total

        train_accuracy = 100 * train_correct / train_total

        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.4f}, accuracy: {train_accuracy:.2f}%")
        print(f"Val loss: {avg_val_loss:.4f}, accuracy: {val_accuracy:.2f}%")


        # Select the best model by validation LOSS, not validation accuracy.
        # This file is weights-only and is used later by the test script.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            #keep the original weights-only file for the test code
            saveCheckpoint(model.state_dict(), best_model_path)

            print("Best model saved!")

        # Save a complete resume checkpoint after each finished epoch, even
        # when the validation loss did not improve.
        numpy_state = np.random.get_state()

        #save the latest completed epoch, even when validation did not improve
        saveCheckpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "best_val_loss": best_val_loss,
                "settings": settings,
                "csv_hashes": csv_hashes,
                
                "python_rng_state": random.getstate(),
                "numpy_rng_state": (
                    numpy_state[0],
                    numpy_state[1].tolist(),
                    int(numpy_state[2]),
                    int(numpy_state[3]),
                    float(numpy_state[4])
                ),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            last_checkpoint_path
        )

        print("Latest checkpoint saved:", last_checkpoint_path, flush=True)

    total_time = time.time() - start_time

    print(f"Training time this session: {total_time / 60:.2f} minutes")

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # --resume continues from the latest full checkpoint by default;
    # --checkpoint selects an explicit saved checkpoint.
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss", choices=["cross_entropy", "geographic"], default="cross_entropy")
    args = parser.parse_args()

    # Apply the same image size and ImageNet normalization as evaluation
    # so the pretrained DINOv2 receives consistently preprocessed inputs.
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # This script runs last3 fine-tuning with 32-image batches.
    # The command line --loss selects geographic loss or cross-entropy.
    trainModel(
        transform,
        fine_tune="last3",
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=32,
        learning_rate=0.0001,
        resume=args.resume or args.checkpoint is not None,
        checkpoint_path=args.checkpoint
    )
