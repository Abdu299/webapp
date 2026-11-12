
from data.paths import CSV_DIR

from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import GeoDataset


# Creates DataLoaders for the training and validation datasets.
# transform: Image preprocessing steps applied to each image.
# num_workers: Number of worker processes used to load data.
# batch_size: Number of images processed together in one batch (default is 32).
def createDataLoaders(transform, num_workers, batch_size=32):

    # Load the training dataset using the training CSV file.
    # GeoDataset reads the image paths and labels from the CSV.
    # The transform is applied to each image when it is loaded.
    train_dataset = GeoDataset(CSV_DIR / "training_split.csv", transform)

    # Load the validation dataset using the validation CSV file.
    # This dataset is used to evaluate the model on data not used for training.
    val_dataset = GeoDataset(CSV_DIR / "validation_split.csv", transform)

    # Create the training DataLoader.
    # batch_size controls how many images are loaded in each batch.
    # shuffle=True randomizes the order of training samples each epoch,
    # helping the model avoid learning from a fixed sample order.
    # num_workers controls how many processes load data in parallel.
    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers )

    # Create the validation DataLoader.
    # shuffle=False keeps the validation samples in their original order,
    # since randomizing their order is not necessary for evaluation.
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # Return both DataLoaders so they can be used by the training code.
    return train_loader, val_loader



if __name__ == "__main__":

    # Define the preprocessing steps applied to every image.
    # Compose combines multiple transformations in the given order.
    transform = transforms.Compose([
        # Resize every image to 224x224 pixels.
        # This gives all images the same dimensions for the ViT model.
        transforms.Resize((224, 224)),

        # Convert the image into a PyTorch tensor.
        # Changes the shape to (channels, height, width)
        # and scales pixel values from 0-255 to 0-1.
        transforms.ToTensor(),

        # Normalize the RGB channels using ImageNet mean and standard deviation.
        # These values are commonly used with pretrained vision models.
        # Normalization helps match the input distribution expected by the model.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create training and validation DataLoaders using the transformations.
    # num_workers=0 means data is loaded in the main process.
    # batch_size is not specified, so the default value of 32 is used.
    train_loader, val_loader = createDataLoaders( transform=transform, num_workers=0)

    # Print the total number of images in each dataset.
    # .dataset accesses the dataset inside the DataLoader.
    # len() returns the number of samples in that dataset.
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Get the first batch from the training DataLoader.
    # iter() creates an iterator, and next() retrieves one batch.
    # images contains the preprocessed image tensors.
    # labels contains the corresponding ground-truth labels.
    images, labels = next(iter(train_loader))

    # Print the shape of the image batch.
    # Normally (32, 3, 224, 224):
    # 32 images, 3 RGB channels, height 224, width 224.
    print("Image shape:", images.shape)

    # Print the shape of the labels.
    # Normally (32,), meaning one label for each image in the batch.
    print("Label shape:", labels.shape)
