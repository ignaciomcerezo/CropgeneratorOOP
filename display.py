def is_ipython():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return True
    except ImportError:
        return False


def display(obj):
    try:
        # Try to use IPython's rich display logic
        from IPython.display import display
        from IPython import get_ipython

        if get_ipython() is not None:
            display(obj)
        else:
            # We are in a script or standard shell
            print(obj)
    except ImportError:
        # IPython isn't even installed
        print(obj)
