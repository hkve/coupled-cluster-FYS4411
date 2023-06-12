import sympy as sp
import drudge
from dummy_spark import SparkContext
import re

ctx = SparkContext()
dr = drudge.PartHoleDrudge(ctx)
dr.full_simplify = False

names = dr.names
i, j, k, l = names.O_dumms[:4]
a, b, c, d = names.V_dumms[:4]

c_ = names.c_
c_dag = names.c_dag

t = sp.IndexedBase("t")
dr.set_dbbar_base(t,2)

T2 = dr.einst(t[a,b,i,j]*c_dag[a]*c_dag[b]*c_[j]*c_[i]/4).simplify()

x = dr.format_latex(T2) + r" + u_{a,b,x,y}"
# \\sum_{.*?}
x = re.sub("\\\sum_{.*?}", "", x)
print(x)
x = re.sub(r"t_{(.*?),(.*?),(.*?),(.*?)}", r"t^{\1\2}_{\3\4}", x)
x = re.sub(r"u_{(.*?),(.*?),(.*?),(.*?)}", r"u^{\1\2}_{\3\4}", x)
print(x)

from sympy.physics.secondquant import simplify_index_permutations

simplify_index_permutations()