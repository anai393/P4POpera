

f = open("Arrowtown.arff", "r")


new_lines = []
with open("Arrowtown.arff") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    for l in lines:
        if len(l) != 0 :
            if l[0] != "@":
                new_l = l.replace('"', '')
                new_lines.append(new_l)
            else:
                new_lines.append(l)


with open('Arrowtown new.arff', 'w') as f:
    for item in new_lines:
        f.write("%s\n" % item)