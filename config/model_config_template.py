class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 64            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    BUFFER_SIZE        = 256           # Buffer Size, used for the shuffling
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 0.998
    DECAY_START        = 20
    REG_FACTOR         = 0.005       # Regularization factor (Used to be 0.005 for the fit mode)

    # Network part
    CHANNELS   = [3, 8, 16, 32, 32, 16]
    NB_BLOCKS  = [1, 2, 2, 2, 1]
    IMAGE_SIZE = 256           # All images will be resized to this size
