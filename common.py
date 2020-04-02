
CLASSES       = ["RM10"]
CLASS = {
    "RM10" : 0
}

CHANNEL     = 3

GRID_WIDTH  = int(154) # The size of 1 cell.
GRID_HEIGHT = int(154)
WIDTH       = int(1080)
HEIGHT      = int(1080)
GRID_X      = WIDTH // GRID_WIDTH # The number of cells.
GRID_Y      = HEIGHT // GRID_HEIGHT

# Inference parameters
DETECTION_PARAMETER = 0.5
NMS = 0.3
