
import math

def ator(a):
    r = math.sqrt(a / math.pi)
    return r

A = [2,4,8,16,32,64,128]
R = [round(ator(a),4) for a in A]
print(R)