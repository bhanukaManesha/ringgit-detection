CLASSES       = ["RM50"]
#CLASSES       = ["RM50","RM100"]
# CLASSES       = ["RM50","RM1","RM10","RM20","RM100"]

CHANNEL     = 3

SAMPLE = 100
BATCH  = 8
EPOCH  = 500

DETECTION_PARAMETER = 0.8

GRID_WIDTH  = int(32) # The size of 1 cell.
GRID_HEIGHT = int(32)
WIDTH       = int(224)
HEIGHT      = int(224)
GRID_X      = WIDTH // GRID_WIDTH # The number of cells.
GRID_Y      = HEIGHT // GRID_HEIGHT


# GRID_WIDTH  = int(8) # The size of 1 cell.
# GRID_HEIGHT = int(8)
# WIDTH       = int(64)
# HEIGHT      = int(64)
# GRID_X      = WIDTH // GRID_WIDTH # The number of cells.
# GRID_Y      = HEIGHT // GRID_HEIGHT


# R_WIDTH       = int(1080)
# R_HEIGHT      = int(1080)
# R_GRID_WIDTH  = int(135) # The size of 1 cell.
# R_GRID_HEIGHT = int(135)
# R_GRID_X      = R_WIDTH // R_GRID_WIDTH # The number of cells.
# R_GRID_Y      = R_HEIGHT // R_GRID_HEIGHT



# R_WIDTH       = int(64)
# R_HEIGHT      = int(64)
# R_GRID_WIDTH  = int(8) # The size of 1 cell.
# R_GRID_HEIGHT = int(8)
# R_GRID_X      = R_WIDTH // R_GRID_WIDTH # The number of cells.
# R_GRID_Y      = R_HEIGHT // R_GRID_HEIGHT