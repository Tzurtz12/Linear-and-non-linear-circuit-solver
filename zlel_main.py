#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Jon Gaminde and Urtzi Jauregi


"""
from zlel import zlel_p1
from zlel import zlel_p2
from zlel import zlel_p3
from zlel import zlel_p3_ondo
from zlel import zlel_p4
from zlel import zlel_p5
import sys

"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/4_zlel_N.cir"
    [cir_el, cir_nd, cir_val, cir_ctrl, analisis] = zl2.cir_parser(filename)
    [cir_el_r, cir_nd_r, cir_val_r,
     cir_ctrl_r, b, n, nodes, el_num, lineal,
     not_lin_elem, dynamic, dyn_elem] = zl4.cir_reshape(
         cir_el, cir_nd, cir_val, cir_ctrl)
    Aa = zl1.incidence_matrix(cir_nd_r, b, n, nodes)
    zl1.check_inc_mat(cir_el, cir_nd, cir_val, cir_ctrl, Aa)
    sol = zl4.do_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num,
                          analisis, Aa, n, b, nodes, lineal, not_lin_elem,
                          dynamic, dyn_elem, filename)
