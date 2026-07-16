def tensor_to_decimal_point(tensor, dp=2):
    return [round(v, ndigits=dp) for v in tensor.tolist()]
