# Pytorch modules with sane defaults for model prototyping
This is a number of pytorch modules (all based on prior work of the ML community) with sane default parameters that I find useful in model prototyping. 

I'll continue to update this repository.

Contains:

* 2d U-Net with different options for how the feature maps are upsampled (to prevent checkerboard artifacts.)
* 3d U-Net
* 2d downsampling network
* 2d upsampling network with different options for how the feature maps are upsampled (to prevent checkerboard artifacts.)
* 2d conv layer that pads to keep the spatial dimensions of the feature map constant, with reflection padding.
