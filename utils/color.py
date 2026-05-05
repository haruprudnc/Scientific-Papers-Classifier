import os

class Color:
    @staticmethod
    def checkSupport():
        # if not sys.stdout.isatty():
        #     return False

        if os.name == 'nt':
            os.system('') 

        return True

    if checkSupport():
        red = "\033[31m"
        green = "\033[32m"
        yellow = "\033[33m"
        blue = "\033[34m"
        cyan = "\033[36m"
        purple = "\033[35m"
        pink = "\033[95m"
        bold = "\033[1m"
        reset = "\033[0m"
        
    else:
        red = green = yellow = blue = cyan = purple = pink = bold = reset = ""