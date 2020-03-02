
CLASSES       = ["RM50","RM10"]
CLASS = {
    "RM50" : 0,
    "RM10" : 1

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
SAMPLE = 1000
EPOCH = 300

# Inference parameters
DETECTION_PARAMETER = 0.8
NMS = 0.5

colors = {
        "RM50" : (0,255,0),
        "RM1" : (255,0,0),
        "RM10" : (0,0,255),
        "RM20" : (0,128,255),
        "RM100" : (255,255,255),
    }
