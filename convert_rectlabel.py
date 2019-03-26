
outfilename = '/Users/pdevine/Documents/Mare/mare yolo annotations/mare_annotations.txt'

with open(outfilename, 'wb') as outfile:
    for filename in glob.glob('/Users/pdevine/Documents/Mare/mare yolo annotations/*.txt'):
        if filename == outfilename:
            # don't want to copy the output into the output
            continue
        with open(filename, 'rb') as readfile:
            print(os.path.basename(filename))