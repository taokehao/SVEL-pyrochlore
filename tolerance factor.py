from mendeleev import element


def print_sample(pred_0list):
    readlines = open("data.txt").readlines()
    for i in pred_0list:
        data = open("data_screen.txt", "a")
        print(readlines[i], file=data, end="")
        data.close()


if __name__ == '__main__':

    # cout = 0
    f = open("data_screen.txt", 'r')
    for line in f.readlines():
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
        atom1 = element(name_list[0])
        atom2 = element(name_list[1])
        Ra25 = 0  # 下面这些定义记录一些信息
        Rb25 = 0
        Ra34 = 0
        Rb34 = 0
        flag_25 = 0
        flag_34 = 0
        Ra_sub_Rb = 0
        for i in atom1.ionic_radii:
            if i.charge == 2 and i.coordination == "VIII":
                flag_25 += 1
                Ra25 = i.ionic_radius
            if i.charge == 3 and i.coordination == "VIII":
                flag_34 += 1
                Ra34 = i.ionic_radius
        for j in atom2.ionic_radii:
            if j.charge == 5 and j.coordination == "VI":
                flag_25 += 1
                Rb25 = j.ionic_radius
            if j.charge == 4 and j.coordination == "VI":
                flag_34 += 1
                Rb34 = j.ionic_radius

        if flag_25 == 2:
            t = 1.43373 - 0.42931 * (Ra25 + 138) / (Rb25 + 138)
            Ra_sub_Rb = Ra25 / Rb25
        elif flag_34 == 2:
            t = 1.43373 - 0.42931 * (Ra34 + 138) / (Rb34 + 138)
            Ra_sub_Rb = Ra34 / Rb34

        if 0.88 <= t <= 0.94:
            if 1.46 <= Ra_sub_Rb <= 1.80:
                # print(t)
                # cout += 1
                # print("计数:", cout)
                # 将筛选完的存起来
                single_line = name_str.replace(" ", "2")
                single_line = single_line + "7\n"
                data = open("fenlei_1.txt", "a")
                print(single_line, file=data, end="")
                data.close()
