"""
Utility functions for XGait model
"""

def get_valid_args(kwargs, targets):
    """Get valid arguments for a function from kwargs"""
    valid_args = {}
    for target in targets:
        if target in kwargs:
            valid_args[target] = kwargs[target]
    return valid_args
