import shutil
from utils.color import Color

def cliProgressBar(current, total, prefix='', suffix='', length=-1):
    if length == -1:
        length = shutil.get_terminal_size().columns
        
    if total == 0: total = 1 
    
    percent = f"{100 * (current / float(total)):.1f}"
    
    text_prefix = f"{prefix} " if prefix else ""
    text_suffix = f" {percent}% | {current}/{total} {suffix}    "
    
    bar_space = length - len(text_prefix) - len(text_suffix)
    bar_space = max(1, bar_space)
    
    filled_blocks = int(bar_space * current // total)
    empty_blocks = bar_space - filled_blocks
    
    bar = f'{Color.blue}█' * filled_blocks + f'{Color.reset}░' * empty_blocks
    
    print(f"\r\033[2K{Color.purple}{text_prefix}{Color.reset}{bar}{Color.reset}{Color.purple}{text_suffix}{Color.reset}", end = "", flush = True)

    if current == total:
        centered_text = "    OPERATION FINISHED!!!!    ".center(length, "=")
        print(f"{Color.green}{centered_text}{Color.reset}")