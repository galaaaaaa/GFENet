# GFE-Net

This project contains the implementation of **GFE-Net**.

## Project Structure

- `gfe_net/`: Core package containing the model implementation.
- `configs/`: Configuration files for training and testing.
- `tools/`: Scripts for training and testing.
## Dependencies
All dependencies are included in ```environment.yml```. To install, run
```
conda env create -f environment.yml
```
## Dataset Preparation
[GrokLST](https://github.com/GrokCV/GrokLST)
The datasets used in our paper are available at:

| Dataset | Description | Download |
| --- | --- | --- |
| GrokLST | Main benchmark dataset for guided LST downscaling. | [GrokLST](https://github.com/GrokCV/GrokLST) |
| ASTER-HLS | Cross-product generalization dataset used for evaluation. | [ASTER-HLS](https://drive.google.com/file/d/1vfGJkXWdBgDUPI-C3VHPl77zHqQpXXqQ/view?usp=drive_link)|



## Usage

### Training

To train the model (e.g., x4 scale):

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 tools/dist_train.sh configs/gfe_net_x4_4xb1-10k_gfe_net.py 1
```

### Testing

To test the model (e.g., x8 scale):

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29501 tools/dist_test.sh configs/gfe_net_x8_4xb1-10k_gfe_net.py work_dirs/gfe_net_x8_4xb1-10k_gfe_net/checkpoint.pth 1
```
##  Acknowledgement
Our code is partially based on this repository: [GrokLST](https://github.com/GrokCV/GrokLST).
