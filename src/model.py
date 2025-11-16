import torch
import torch.nn as nn
import timm


def create_model(model_name="efficientnet_b3", num_classes=5, pretrained=True):
    """
    Create an image classification model using timm EfficientNet models.
    """

    # Load EfficientNet backbone
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


if __name__ == "__main__":
    # Test model creation
    m = create_model("efficientnet_b3", num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Output shape:", y.shape)
