
CLASSES       = ["RM50","RM100"]
CLASS = {
    "RM50" : 0,
    "RM100" : 1
}
CHANNEL     = 3

GRID_WIDTH  = int(16)
GRID_HEIGHT = int(16)
WIDTH       = int(112)
HEIGHT      = int(112)
GRID_X      = WIDTH // GRID_WIDTH
GRID_Y      = HEIGHT // GRID_HEIGHT

RWIDTH = int(378)
RHEIGHT = int(378)

# Training parameters
SAMPLE = 100
BATCH  = 8
EPOCH  = 200

# Inference parameters
DETECTION_PARAMETER = 0.5
NMS = 0.5


# Generator parameters
money_path = "cash"
validation_path = "data/val/"
test_path = "data/test/"

# CLASS_INDEX = ["RM50", "RM1", "RM10", "RM20","RM100"]
