import sympy as sp
import drudge
from dummy_spark import SparkContext

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

ham = dr.ham

# CCD truncates after 2x commutator
c1 = (ham | T2).simplify()
c2 = (c1 | T2).simplify()
# c3 = (c2 | T2).simplify()
# c4 = (c3 | T2).simplify()

h_bar = (ham + c1 + sp.Rational(1, 2)*c2) #+ sp.Rational(1, 6)*c3 + sp.Rational(1, 24) * c4).simplify()

h_bar.repartition(cache=True)

Energy_equation = h_bar.eval_fermi_vev().simplify()

p2h2 = c_dag[i]*c_dag[j]*c_[b]*c_[a]
Amplitude_equation2 = (p2h2 * h_bar).eval_fermi_vev().simplify()

with dr.report("out/ccd_reduced.html", "CCD equations") as rep:
    rep.add("Energy equation", Energy_equation)
    rep.add("2p2h amplitude equation", Amplitude_equation2)