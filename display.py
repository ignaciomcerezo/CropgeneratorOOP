from PIL.Image import Image

def display(thing_to_display, **kwargs):
    if isinstance(thing_to_display, Image):
        thing_to_display.show()
    else:
        print("<not iPython> ", end ="")
        print(thing_to_display, **kwargs)
