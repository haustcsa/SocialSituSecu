# _*__coding:utf-8 _*__
# @Time :2022/9/29 0029 11:08
# @Author :bay
# @File figure3-result_compute_pic.py
# @Software : PyCharm
bds = [30, 31, 32, 33, 36, 35, 35, 38]
bcs = [20, 22, 23, 22, 24, 23, 23, 24]
ps = []
rs = []
fs = []
for bd, bc in zip(bds, bcs):
    p = bc/bd
    r = bc/30
    f = (2*p*r)/(p+r)
    ps.append(round(p, 5))
    rs.append(round(r, 5))
    fs.append(round(f, 5))
print("准确率", ps)
print("召回率", rs)
print("F1值", fs)