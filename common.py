CLASSES       = ["RM50"]
#CLASSES       = ["RM50","RM100"]
# CLASSES       = ["RM50","RM1","RM10","RM20","RM100"]

CHANNEL     = 3

SAMPLE = 128
BATCH  = 8
EPOCH  = 100

DETECTION_PARAMETER = 0.8

GRID_WIDTH  = int(8) # The size of 1 cell.
GRID_HEIGHT = int(8)
WIDTH       = int(64)
HEIGHT      = int(64)
GRID_X      = WIDTH // GRID_WIDTH # The number of cells.
GRID_Y      = HEIGHT // GRID_HEIGHT

MONEY_PATH = "images/"
LOCATION_X = 0.5
LOCATION_Y = 0.5


CLASS_INDEX = ["RM50", "RM1", "RM10", "RM20","RM100"]

CLASS = {
    "RM50" : "0",
}