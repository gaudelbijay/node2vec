import numpy as np

def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept,alias = [0]*l,[0]*l
    small,large = [],[]
    for i,val in enumerate(area_ratio):
        if val<1:
            small.append(i)
        else: large.append(i)

    while small and large:
        small_idx,large_idx = small.pop(),large.pop()
        accept[small_idx]=area_ratio[small_idx]
        alias[small_idx]=large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
        if area_ratio[large_idx]<1.0:
            small.append(large_idx)
        else: large.append(large_idx)
    
    while small:
        small_idx = small.pop()
        accept[small_idx]=1

    while large:
        large_idx = large.pop()
        accept[large_idx]=1
        
    return accept,alias

def alias_sample(accept,alias):
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

            