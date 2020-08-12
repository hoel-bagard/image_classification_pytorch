class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 32            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    BUFFER_SIZE        = 256           # Buffer Size, used for the shuffling
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 5e-3
    DECAY_START        = 100
    REG_FACTOR         = 0.00005       # Regularization factor (Used to be 0.005 for the fit mode)

    # Network part
    CHANNELS   = [16, 32, 64, 128, 256]
    IMAGE_SIZE = 256           # All images will be resized to this size
