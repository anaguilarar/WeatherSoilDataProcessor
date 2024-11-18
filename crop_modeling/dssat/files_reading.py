
## taken from https://github.com/AgroClimaticTools/dssat-pylib/blob/main/dssatpylib/util_read_dssat_out.py
def delimitate_header_indices(section_header_str):
    start_indices = [0]+[i for i, character in enumerate(section_header_str)
                           if character == ' ' and section_header_str[i+1] != ' ']
    end_indices = start_indices[1:] + [len(section_header_str)+20]
    return list(zip(start_indices, end_indices))

def section_indices(lines, pattern = '@'):
    #with open(path, 'r', encoding="utf-8") as fn:
    for i, line in enumerate(lines):
        stiped_line = line.strip()
        if stiped_line.startswith(pattern):
            yield i
            