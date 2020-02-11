CLASSES       = ["RM50"]
CLASS = {
    "RM50" : "0",
}
CHANNEL     = 3

GRID_WIDTH  = int(135) # The size of 1 cell.
GRID_HEIGHT = int(135)
WIDTH       = int(1080)
HEIGHT      = int(1080)
GRID_X      = WIDTH // GRID_WIDTH # The number of cells.
GRID_Y      = HEIGHT // GRID_HEIGHT

# Inference parameters
DETECTION_PARAMETER = 0.5
NMS = 0.1
