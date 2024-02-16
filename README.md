# DentSeg / Dental Segmentation with a Flexible U-Net

## Description

This project showcases the segmentation of dental X-ray imagery using a PyTorch implementation of a flexible U-Net architecture. It introduces the capability to operate in Half U-Net mode, as detailed in the paper ["Half U-Net: A Simplified U-Net Architecture for Medical Image Segmentation"](https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full). Additionally, it incorporates variations of this structure and integrates the Ghost Module v2 concept from ["GhostNet: More Features from Cheap Operations"](https://paperswithcode.com/method/ghost-module) and "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," aimed at creating extra feature layers with minimal compute requirements.

## Features

- **Flexible U-Net Architecture**: Adjusts to Half U-Net mode for efficient computation while maintaining segmentation performance.
- **Ghost Module Integration**: Utilizes "cheap operations" to generate additional feature layers, maintaining the model's capability while making computational savings.
- **Configurable Channels**: Offers the option to fix channels throughout the U-Net, adhering to the methodologies proposed by the Half U-Net paper.

## Installation

To install and run the Dental Segmentation project, follow these steps:

1. Clone the repository to your local machine:
```
git clone https://github.com/yourgithubusername/dental-segmentation-flexible-unet.git
```

2. Navigate to the project directory:
```
d dental-segmentation-flexible-unet
```

3. Install the required dependencies. Ensure you have Python and PyTorch installed, then install the other dependencies as detailed in `dentsegdataset.py` and `HalfUNet` files.

## Usage

After installation, the project can be run from the command line for processing dental X-ray images. Detailed usage instructions and command-line interface (CLI) options will be provided, allowing users to specify model checkpoints and configuration settings.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was based on the architecture from the papers on Half U-Net architecture and GhostNet technologies. 
