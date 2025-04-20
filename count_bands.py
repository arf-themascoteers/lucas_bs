import final_selected_bands as fsb

a_count = []
b_count = []

for keys in fsb.bsdr_bands_as_list.keys():
    a_count.append(len(fsb.selected_bands_as_list[keys]))
    b_count.append(len(fsb.bsdr_bands_as_list[keys]))

print(a_count)
print(b_count)