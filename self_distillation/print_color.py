import inspect

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def pc(*kwarg, c = "y"):
    if c == "y":
        color = bcolors.WARNING
    if c == "b":
        color = bcolors.OKBLUE
    if c == "g":
        color = bcolors.OKGREEN
        
    print(color, *kwarg, bcolors.ENDC)

def ps(*tensors):
    for tensor in tensors:
        print(bcolors.WARNING, _get_variable_name(tensor), tensor.shape, bcolors.ENDC)

def _get_variable_name(var):
    frame = inspect.currentframe().f_back
    variable_names = {id(v): k for k, v in frame.f_locals.items()}
    return variable_names.get(id(var), None)
