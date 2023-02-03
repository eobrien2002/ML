from itertools import zip_longest
import re


data_path = "weather.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

lines = [re.sub(r"(?:\@|https?\://)\S+", "", line).strip() for line in lines]
lines = [re.sub(r"#", "", x) for x in lines]
lines = [item for item in lines if not item.isdigit()]
lines = [item for item in lines if item != '']

# group lines by response pair

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
pairs = list(grouper(lines, 2))
