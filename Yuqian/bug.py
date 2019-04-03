def is_shift(l1,l2):
    if len(l1)!= len(l2):
        return False
    ls = len(l1)
    for i in range(ls):
        tmp = l1[i:] + [128 + x for x in l1[:i]]
        tmp_set = []
        for i, x in enumerate(tmp):
            if x - l2[i] not in tmp_set:
                tmp_set.append(x-l2[i])
        if len(tmp_set) == 1:
            return True
    return False

if __name__ == "__main__":
    print(is_shift([1, 5, 11, 14, 122], [11, 18, 22, 28, 31]))
