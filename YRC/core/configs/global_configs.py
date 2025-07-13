GLOBAL_VARIABLES = {}


def set_global_variable(key, value):
    global GLOBAL_VARIABLES
    GLOBAL_VARIABLES[key] = value


def get_global_variable(key):
    return GLOBAL_VARIABLES.get(key)


def get_all_global_variables():
    return GLOBAL_VARIABLES
