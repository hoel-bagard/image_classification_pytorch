# Image Classification using PyTorch
## Installation

### Requirements/dependencies
- Python >=3.10
- PyTorch (preferably with GPU)
- Packages from the `requirements.txt` file

<details>
<summary>Example (virtualenv/pip)</summary>
Assuming you're on a linux PC/server with Python>=3.10 and PyTorch already installed, you can use those commands:

```
virtualenv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```
</details>

### Clone the repository
```
git clone git@github.com:hoel-bagard/image_classification_pytorch.git --recurse-submodules
```

### Get some data and format it:

You need to split the images between two folders: "Train" and "Validation" (the names are hard coded).
You then need to create a `classes.names` file next to the Train and Validation folders, with the names of the classes (one per line).

<details>
  <summary>CIFAR-10 example</summary>

The commands below will download, extract and format the cifar 10 dataset into the `./data/cifar_10_images` folder.

```
wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz -P data
tar -xvf data/cifar-10-python.tar.gz -C data
python utils/cifar_10.py data/cifar-10-batches-py
rm data/cifar-10-python.tar.gz
rm -r data/cifar-10-batches-py/
```

Note:
You'll need to modify a few values in `config/model_config.py` in the next step since cifar10's images are small.
```python
    CROP_IMAGE_SIZES: tuple[int, int] = (32, 32)  # Center crop
    RESIZE_IMAGE_SIZES: tuple[int, int] = (32, 32)  # All images will be resized to this size
...
    CHANNELS: list[int] = field(default_factory=lambda: [3, 16, 32, 16])
    SIZES: list[int | tuple[int, int]] = field(default_factory=lambda: [3, 3, 3])   # Kernel sizes
    STRIDES: list[int | tuple[int, int]] = field(default_factory=lambda: [2, 2, 2])
    PADDINGS: list[int | tuple[int, int]] = field(default_factory=lambda: [1, 1, 1])
    BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 1])
```
</details>


## Config files
In the config folder of this repo you will find two config template files. You need to copy them and remove the "_template" part like this:
```
cp config/data_config_template.py config/data_config.py
cp config/model_config_template.py config/model_config.py
```

### DataConfig
Contains most of the parameters regarding the data. Most of the values in the template can be kept as they are. The 3 paths usually need to be modified for each training (`DATA_PATH`, `CHECKPOINT_DIR` & `TB_DIR`). 

### ModelConfig
Contains the parameters that influence training. Most default values should work okayish, but you'll need to modify a few:
- `MAX_EPOCHS`: usually around 400 or 600 epochs is enough, you will need to train at least once to get an idea for your particular dataset.
- `IMG_MEAN` and `IMG_STD`: The defaults are the imagenet ones. You can keep them as long as they are not too different from the actual ones (especially if using a pretrained model).

## Train
Once you have the environment all set up and your two config files ready, training an AI is straightforward. Just run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Results

TODO: TB screenshots

## Inference

TODO

### Misc
#### Formating
The code is trying to follow diverse PEPs conventions (notably PEP8). To have a similar dev environment you can install the following packages (pacman is for arch-based linux distros):

```
sudo pacman -S flake8 python-flake8-docstrings
pip install pep8-naming flake8-import-order
```

#### Typing
Typing is done using Pyright.\
The type stubs for OpenCV are taken from [this repo](https://github.com/microsoft/python-type-stubs/tree/main/cv2).
