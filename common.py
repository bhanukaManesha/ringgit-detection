
CLASSES       = ["RM10"]
CLASS = {
    "RM10" : 0,
}
CHANNEL     = 3

GRID_WIDTH  = int(32)
GRID_HEIGHT = int(32)
WIDTH       = int(224)
HEIGHT      = int(224)
GRID_X      = WIDTH // GRID_WIDTH
GRID_Y      = HEIGHT // GRID_HEIGHT

RWIDTH = int(378)
RHEIGHT = int(378)

# Training parameters
SAMPLE = 16000
EPOCH = 15
THREADS = 4

# Inference parameters
DETECTION_PARAMETER = 0.75
NMS = 0.5

colors = {
        "RM50" : (0,255,0),
        "RM1" : (255,0,0),
        "RM10" : (0,0,255),
        "RM20" : (0,128,255),
        "RM100" : (255,255,255),
    }
