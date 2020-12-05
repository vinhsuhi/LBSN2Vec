
for data in ["hongzhi", "Jakarta", "Istanbul", "NYC", "TKY", "SaoPaulo", "KualaLampur"]:
    base = "edgelist_{}".format(data)

    file1 = open("{}_0".format(base), 'r')
    file2 = open("{}_1".format(base), 'r')
    file3 = open("{}_2".format(base), 'r')
    file4 = open("{}_3".format(base), 'r')

    lines1 = [int(line.split()[0]) for line in file1]
    # lines1 = [int(line.split()[0]) for line in file1]
    # lines1 = [int(line.split()[0]) for line in file1]
    # lines1 = [int(line.split()[0]) for line in file1]
    # lines2 = [line for line in file2]
    # lines3 = [line for line in file3]
    # lines4 = [line for line in file4]
    print(data, len(lines1), max(lines1))


    # print(lines1 == lines2)
    # print(lines3 == lines2)
    # print(lines4 == lines3)
# print(lines1 == lines2)