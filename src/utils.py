# Find the number of levels automatically
def find_level(inputs):
    current = inputs[0]
    level = -1
    while isinstance(current, list):
        current = current[0]
        level += 1

    return level
