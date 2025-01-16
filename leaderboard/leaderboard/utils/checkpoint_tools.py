import json
import os


def fetch_dict(endpoint):
    data = {}
    if os.path.exists(endpoint):
        with open(endpoint) as fd:
            try:
                data = json.load(fd)
            except json.JSONDecodeError:
                data = {}

    return data

def save_dict(endpoint, data):
    with open(endpoint, 'w') as fd:
        json.dump(data, fd, indent=4)

def save_array(endpoint, data):
    data.dump(endpoint)

def save_array_txt(endpoint, data):

    with open(endpoint, 'w') as f:

        line_str = '# This is only for manual checking, the automatically imported file is the .dat one\n'
        x_size, y_size, _ = data.shape

        line_str += "[\n    "
        for x in range(x_size):
            line_str += "["
            for y in range(y_size):
                line_str += "[%7.3f, %7.3f, %5.3f, %4.1f], " % (tuple(data[x,y]))

            line_str = line_str[:-2]
            line_str += "],\n    "

        line_str = line_str[:-4]
        line_str += "]"

        f.write(line_str)