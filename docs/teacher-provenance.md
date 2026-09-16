# The teacher uses torchvision's ImageNet weights.

The teacher uses MobileNet_V3_Small_Weights.IMAGENET1K_V1 from torchvision.
The official PyTorch URL supplies the weights at https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth.
The published filename and installed torchvision source provide the SHA256 prefix 047dcff4.
The downloaded file matches that independent prefix.
The configuration records the full observed SHA256 to verify subsequent local use.
The full digest is a local integrity record; the independent publisher check covers the published prefix.

The [torchvision source license](https://github.com/pytorch/vision/blob/main/LICENSE) covers the library under BSD-3-Clause.
The [torchvision pretrained model notice](https://github.com/pytorch/vision#pre-trained-model-license) states that model terms may depend on the training dataset.
The teacher was trained on ImageNet-1K.
This project uses the weights for local noncommercial research and does not redistribute them.
This record does not grant commercial model or ImageNet rights.
Production qualification requires a separate rights review.
