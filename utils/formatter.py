# string -> bool
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# file format(img, video) checker
def fileformat(path):
    if path.split('.')[1] in ['jpg', 'jpeg', 'png']:
        return 0
    elif path.split('.')[1] in ['mp4', 'avi', 'mkv']:
        return 1
    else:
        return -1

def optionChecker(option):
    optDict = {'skel': False, 'keyp': False, 'label': False}
    if 's' in option:
        optDict['skel'] = True
    if 'k' in option:
        optDict['keyp'] = True
    if 'l' in option:
        optDict['label'] = True

    return optDict