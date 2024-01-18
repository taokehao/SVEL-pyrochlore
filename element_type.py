# 筛选出有毒元素
readlines = open("fenlei_1_2.txt").readlines()
element = []
# Youdu = ['Ru', 'Rh', 'Re', 'Os', 'Ir', 'Tc', 'Ta', 'W', 'Pt', 'Au', 'Sb', 'Cd', 'Hg', 'Eu', 'Pa', 'U', 'Np', 'Pu',
# 'Yb', 'Pr', 'Sc', 'Y', 'Tl', 'Cr', 'Pd', 'La', 'Ce', 'Pm', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu']

# 封存于2023.10.13
# Youdu = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Sc',
#          'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
#          'Au', 'Ag', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt',
#          'Li', 'Cs', 'Be', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Mo', 'W', 'Ga', 'In', 'Tl', 'Ge', 'Re', 'Se', 'Te', 'Sc', 'Y', 'Fr', 'Ra', 'Po', 'Tc', 'Pm']

Youdu = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Sc',
         'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
         'Au', 'Ag', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt', 'Re', 'Sb', 'In', 'Sn',
         'Tl', 'Bi', 'Tc', 'Mo']

# Youdu = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Sc',
#          'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
#          'Au', 'Ag', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt',
#          'Cs', 'Be', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Mo', 'W', 'Ga', 'In', 'Tl', 'Ge', 'Re', 'Se', 'Te', 'Sc', 'Y', 'Fr', 'Ra', 'Po', 'Tc', 'Pm']

for line in readlines:
    line = line.strip("\n")
    line = line.strip("7")
    line = line.replace("2", " ")
    name_str = line

    name_list = []  # 记录每个原子的列表
    atom_count = 0  # 原子个数
    name_num = 0  # 循环次数
    for i in name_str:  # 得到输入中各个原子的循环
        if i != " ":
            if name_num == 0 or name_str[name_num - 1] == " ":
                name_list.append(i)
                atom_count += 1
            else:
                name_list[atom_count - 1] = name_list[atom_count - 1] + i
        name_num += 1
    # print(name_list)
    for i in name_list:
        if i not in element:
            element.append(i)
    if name_list[0] not in Youdu and name_list[1] not in Youdu:
        single_line = name_str.replace(" ", "2")
        single_line = single_line + "7"
        print(single_line)

print("140种物质中的元素种类:", len(element))
print(element)
for i in element:
    if i not in Youdu:
        print(i, end="")
        print(', ', end="")
