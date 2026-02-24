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

You can find the required data from the link [GrokLST](https://github.com/GrokCV/GrokLST) and place them according to the path shown in ```configs/_base_/datasets```.

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
