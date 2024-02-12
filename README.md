# Image Classification using PyTorch

## Installation
### Requirements
- Python >=3.10
- Poetry

### Install
If simply using the package on cpu:
```console
poetry install
```

If developing you can add any of the following options:
```console
poetry install --with dev,test,gpu
```

Then use `poetry shell` to enter the virtualenv.

## Data
### Get some data and format it:

You need to split the images into a validation and a train folders.
For each class, place all the images in a folder with the class's name.
You then need to create a `classes.names` file next to the train and validation folders, with the names of the classes (one per line).

<details>
  <summary>Structure example</summary>
cifar-10/
├── Train/
│   ├── airplaine
│   ├── automobile
│   ├── bird
│   ├── cat
│   └── ...
├── Validation/
│   ├── airplaine
│   ├── automobile
│   ├── bird
│   ├── cat
│   └── ...
└── classes.names
</details>

<details>
  <summary>CIFAR-10 instructions</summary>

The commands below will download, extract and format the cifar 10 dataset into the `./data/cifar_10_images` folder.

```console
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

<details>
  <summary>Imagenette instructions</summary>

The commands below will download, extract and format the cifar 10 dataset into the `./data/cifar_10_images` folder.

```console
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz -P data
tar -xvf data/imagenette2.tgz -C data
python utils/preprocess_imagenette.py data/imagenette2
rm data/imagenette2.tgz
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

<details>
  <summary>Cifar-10 example</summary>

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

## Train
Once you have the environment all set up and your two config files ready, training an AI is straightforward. Just run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Results

TODO: TB screenshots

## Inference

TODO
