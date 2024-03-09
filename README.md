# DentSeg / Dental Segmentation with a Flexible U-Net

## Description

This project showcases the segmentation of dental X-ray imagery using a PyTorch implementation of a flexible U-Net architecture. It introduces the capability to operate in Half U-Net mode, as detailed in the paper ["Half U-Net: A Simplified U-Net Architecture for Medical Image Segmentation"](https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full). Additionally, it incorporates variations of this structure and integrates the Ghost Module v2 concept from ["GhostNet: More Features from Cheap Operations"](https://paperswithcode.com/method/ghost-module) and "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," aimed at creating extra feature layers with minimal compute requirements.

## Features

- **Flexible U-Net Architecture**: Adjusts to Half U-Net mode for efficient computation while maintaining segmentation performance.
- **Ghost Module Integration**: Utilizes "cheap operations" to generate additional feature layers, maintaining the model's capability while making computational savings.
- **Configurable Channels**: Offers the option to fix channels throughout the U-Net, adhering to the methodologies proposed by the Half U-Net paper.


## Dataset

The dental X-ray image dataset used in training was sourced from [Humans in the Loop Dental x-ray imagery](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images)

A slimmed down [archive](dentseg_dataset.tar.gz) of the dataset is provided. 


## Installation

### From Source

1. Clone the repository to your local machine:
```
git clone https://github.com/alyxking/dentseg.git
```

2. Navigate to the project directory:
```
cd dentseg
```

3. Ensure required dependencies as detailed in `requirements.txt` are installed. Or, run from the container environment as follows.

### From Container

1. Pull the docker image
```
docker pull ghcr.io/alyxking/dentseg:tag
```
2. Extract the dataset archive

2. Run the container (set DATASET_HOST_PATH and SRC_HOST_PATH)
```
docker run --rm --gpus all \
  -v "DATASET_HOST_PATH:/app/dataset" \
  -v "SRC_HOST_PATH:/app/src" \
  -p "80:80" \
  -p "8888:8888" \
  --name dentseg dentseg \
  jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

## Usage

Cloned repo: Extract the dataset archive and designate the path in dentsegdataset.py

From container:
CLI usage is WIP. Run from the Jupyter Notebook 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was based on the architecture from the papers on Half U-Net architecture and GhostNet technologies. Training was performed on the Humans in the Loop dental x-ray imagery dataset.

## References

```
@dataset{HumansInTheLoop2023,
  author       = {Humans In The Loop},
  title        = {{Teeth Segmentation on dental X-ray images}},
  year         = 2023,
  publisher    = {Kaggle},
  version      = {1},
  doi          = {10.34740/KAGGLE/DSV/5884500},
  url          = {https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-xray-images}
}
```
```
@article{LuHaoran2022,
  author       = {Lu Haoran and She Yifei and Tie Jun and Xu Shengzhou},
  title        = {{Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation}},
  journal      = {Frontiers in Neuroinformatics},
  volume       = {16},
  year         = 2022,
  doi          = {10.3389/fninf.2022.911679},
  url          = {https://www.frontiersin.org/articles/10.3389/fninf.2022.911679},
  issn         = {1662-5196}
}
```
```
@misc{Han2020GhostNet,
  author       = {Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
  title        = {{GhostNet: More Features from Cheap Operations}},
  year         = 2020,
  eprint       = {1911.11907},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/1911.11907}
}
```
