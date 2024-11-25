
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
            
def join_row_using_header_indices(section_header_str, section_line_str, row_to_replace):
    
    header_indices = delimitate_header_indices(section_header_str)

    stline =[None]*len(header_indices) 
    for z, (i, j) in enumerate(header_indices):
        stline[z] = ' '*len(section_header_str)

        if z == 0:
            stline[z] = section_line_str[:(j-i)]
        elif section_header_str[i:j].count(' ')>0:
            posini = [pos for pos, char in enumerate(section_header_str[i:j]) if char == ' ']
            stline[z] = ' '* (posini[0]+1) + row_to_replace[z][:(j-i)] + ' '* abs((len(row_to_replace[z])+1)-(j-i))  
        else:
            stline[z] = row_to_replace[z][:(j-i)] 

    return ''.join(stline)+'\n'

def getting_line_inoutputfile(header, line):
    """
    this only works assuming that the variables are not space separated, and the only one is the initial planting
    """
    
    dataline = [i for i in line.split(' ') if i != '']
    newline = [' ']
    count = 0
    while len(newline) < (len(header)) and count < len(dataline):
        if dataline[count].startswith('Initial'):
            newline.append((dataline[count] + ' ' + dataline[count + 1]).strip())
            count = count + 2
            continue
        elif dataline[count].startswith('Planting'):
            newline.append((dataline[count] + ' '+ dataline[count + 1] + ' ' + dataline[count + 2]).strip())
            count = count + 3
            continue
        else:
            newline.append(dataline[count].strip())
            count +=1
    
    return newline
