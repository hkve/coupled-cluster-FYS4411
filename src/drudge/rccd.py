import sympy as sp
import drudge
from dummy_spark import SparkContext

ctx = SparkContext()
dr = drudge.RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

names = dr.names
e_ = names.e_
i, j, k, l = names.O_dumms[:4]
a, b, c, d = names.V_dumms[:4]

t = sp.IndexedBase("t")
dr.set_n_body_base(t, 2)

T2 = dr.einst(
    sp.Rational(1,2) * t[a,b,i,j] * e_[a, i] * e_[b, j]
).simplify()
T2.cache()

ham = dr.ham

c1 = (ham | T2).simplify()
c2 = (c1 | T2).simplify()
c3 = (c2 | T2).simplify()
c4 = (c3 | T2).simplify()

h_bar = (ham + c1 + sp.Rational(1,2)*c2 + sp.Rational(1,3)*c3 + sp.Rational(1,4)*c4).simplify()

Energy_equation = h_bar.eval_fermi_vev().simplify()

p2h2 = e_[i,a] * e_[j,b]
Amplitude_equation = (p2h2*h_bar).eval_fermi_vev().simplify()

with dr.report("out/rccd.html", "Restricted CCD equations") as rep:
    rep.add("Energy equation", Energy_equation)
    rep.add("2p2h amplitude equation", Amplitude_equation)