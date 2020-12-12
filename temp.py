def most_significant_1_bit_table(bit: np.uint64) -> np.uint64:
    if bit > np.uint64(127): return np.uint64(7)
    if bit > np.uint64(63):  return np.uint64(6)
    if bit > np.uint64(31):  return np.uint64(5)
    if bit > np.uint64(15):  return np.uint64(4)
    if bit > np.uint64(7):   return np.uint64(3)
    if bit > np.uint64(1):   return np.uint64(1)
    return np.uint64(0)
