#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


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
        filename = "cirs/opamp/0_zlel_OPAMP.cir"
    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctrl] = zl1.cir_parser(filename)
    # zl1.cir_parser(filename)
    [cir_el_r, cir_nd_r, b, n, nodes, el_num] = zl1.cir_reshape(cir_el, cir_nd,
                                                                cir_val,
                                                                cir_ctrl)
    zl1.print_cir_info(cir_el_r, cir_nd_r, b, n, nodes, el_num)
    zl1.print_incidence_matrix(cir_nd_r, b, n, nodes)

    # THIS FUNCTION IS NOT COMPLETE
