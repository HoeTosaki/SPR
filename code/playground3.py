# 快速排序算法实现
def sort_quick(lst):
    return r_sort_quick(lst, 0, len(lst) - 1)


def r_sort_quick(lst, st, ed):
    if st >= ed:
        return lst
    pivot = partition(lst, st, ed)
    r_sort_quick(lst, st, pivot - 1)
    r_sort_quick(lst, pivot + 1, ed)
    return lst


def partition(lst, st, ed):
    pivot = st
    p_left = st + 1
    p_right = ed
    while p_left <= p_right:
        while p_left <= ed and lst[p_left] <= lst[pivot]:
            p_left += 1
        while p_right >= st + 1 and lst[p_right] >= lst[pivot]:
            p_right -= 1
        if p_left > ed or p_right < st + 1:
            break
        if p_left >= p_right:
            break
        if lst[p_left] > lst[p_right]:
            lst[p_left], lst[p_right] = lst[p_right], lst[p_left]
    lst[p_right], lst[pivot] = lst[pivot], lst[p_right]
    pivot = p_right
    return pivot


def sort_bubble(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst)-i-1):
            if lst[j] > lst[j+1]:
                lst[j],lst[j+1] = lst[j+1],lst[j]
    return lst

if __name__ == '__main__':
    print(sort_bubble([1, 5, 2, 3, 9, 15, 11]))