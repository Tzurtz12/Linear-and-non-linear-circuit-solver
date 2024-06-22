#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import time
import math
import numpy as np
import sys

if __name__ == "zlel.zlel_p5":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
    import zlel.zlel_p3 as zl3
    import zlel.zlel_p4 as zl4
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2
    import zlel_p3 as zl3
    import zlel_p4 as zl4


def cir_reshape(cir_el, cir_nd, cir_val, cir_ctrl):
    """
        This time, we need to include the dynamic elements, capacitor and
        inductor. To do so, we create an array to save the indices of dynamic
        elements and also create a logical variable that states wheter there
        are any dynamic elements. It's quite similar to what we did before with
        the non lineal elements.

    Args:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size(b,3)
        cir_ctrl: np array of strings with the element which branch
        controls the controlled sources. size(1,b)

    Returns:
        cir_el_r: reshaped cir_el
        cir_nd_r: reshaped cir_nd. Now it will be a (b,2) matrix
        b: # of branches
        n: # number of nodes
        nodes: an array with the circuit nodes sorted
        el_num:  the # of elements.

    """

    el_num = len(cir_el)

    cir_el_r = np.empty([0], dtype=str)
    cir_nd_r = np.empty([0, 2], dtype=int)
    cir_val_r = np.empty([0, 3], dtype=float)
    cir_ctrl_r = np.empty([0], dtype=str)

    lineal = True
    not_lin_elem = np.empty([0], dtype=int)
    dynamic = False
    dyn_elem = np.empty([0], dtype=int)

    for i in range(el_num):
        if 'a' in cir_el[i].lower():
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_in")
            in_nd = np.array([[cir_nd[i, 0], cir_nd[i, 1]]])
            cir_nd_r = np.append(cir_nd_r, in_nd, axis=0)
            val = np.array([[0, 0, 0]])
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, '0')
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_ou")
            out_nd = np.array([[cir_nd[i, 2], cir_nd[i, 3]]])
            cir_nd_r = np.append(cir_nd_r, out_nd, axis=0)
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, '0')
        elif 'q' in cir_el[i].lower():
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_be")
            be_nd = np.array([[cir_nd[i, 1], cir_nd[i, 2]]])
            cir_nd_r = np.append(cir_nd_r, be_nd, axis=0)
            val = np.array([cir_val[i]])
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, '0')
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_bc")
            bc_nd = np.array([[cir_nd[i, 1], cir_nd[i, 0]]])
            cir_nd_r = np.append(cir_nd_r, bc_nd, axis=0)
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, '0')
        else:
            cir_el_r = np.append(cir_el_r, cir_el[i])
            nd = np.array([cir_nd[i][0:2]])
            cir_nd_r = np.append(cir_nd_r, nd, axis=0)
            val = np.array([cir_val[i]])
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, cir_ctrl[i])
    b = len(cir_el_r)
    for i in range(b):
        if 'q' in cir_el_r[i].lower():
            lineal = False
            not_lin_elem = np.append(not_lin_elem, [i], axis=0)
        elif 'd' in cir_el_r[i].lower():
            lineal = False
            not_lin_elem = np.append(not_lin_elem, [i], axis=0)
        elif 'c' in cir_el_r[i].lower():
            dynamic = True
            dyn_elem = np.append(dyn_elem, [i], axis=0)
        elif 'l' in cir_el_r[i].lower():
            dynamic = True
            dyn_elem = np.append(dyn_elem, [i], axis=0)

    nodes = np.unique(cir_nd_r)
    # print(not_lin_elem)
    # print(cir_el_r)
    if 0 not in nodes:
        raise NoReferenceNode('Reference node "0" is not defined in the '
                              'circuit.')
    n = len(nodes)
    return (cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, n, nodes,
            el_num, lineal, not_lin_elem, dynamic, dyn_elem)


def do_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num, analisis,
                Aa, n, b, nodes, lineal, not_lin_elem, filename):
    A_r = zl1.reduce_incidence_matrix(Aa)
    for f in analisis:
        if f[0].lower() == '.pr':
            zl1.print_cir_info(cir_el_r, cir_nd_r, b, n, nodes, el_num)
            # zl1.print_incidence_matrix(cir_nd_r, b, n, nodes)
            print('\nIncidence Matrix:')
            print(Aa)
        elif f[0].lower() == '.op':
            if lineal:
                sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                                  b, A_r, n, 0, 0)
            else:
                sol = zl3.NR(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                             b, n, A_r, nodes, el_num, not_lin_elem)
            zl2.print_solution(sol, b, n)
        elif f[0].lower() == '.dc':
            dc_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                            A_r, n, b, f, filename)
            source = f[8]
            filename = filename[:-4] + '_' + source + ".dc"
            # save_as_csv(b, n, filename)
            zl2.plot_from_cvs(filename, "v", "e1", "")
        elif f[0].lower() == '.tr':
            tr_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num,
                        A_r, n, b, f, lineal, not_lin_elem, filename)
            filename = filename[:-3] + "tr"
            # save_as_csv(b, n, filename)
            zl2.plot_from_cvs(filename, "t", "e1", "")


def elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, A, n,
                Vj, h, t=None):
    '''
    We now include the equations for capacitors and inductors, aplying the
    Euler backward:
        Capacitor --> math:'v_{c,k+1}-\\frac{h}{C}i_{c,k+1}=v_{c,k}'
        Inductor --> math:'\\frac{h}{L}v_{c,k+1}-i_{c,k+1}=v_{c,k}'

    Parameters
    ----------
    cir_el_r : reshaped cir_el.
    cir_nd_r : reshaped cir_nd.
    cir_val_r : reshaped cir_val.
    cir_ctrl_r : reshaped cir_ctrl.
    b : number of branches.
    A : incidence matrix.
    n : number of nodes.
    Vj : initial voltage for discrete analisis.
    h : step for Euler backward.
    t : variable of time.

    Returns
    -------
    emaitza_m : solution of the Tableau equations.

    '''
    n -= 1
    M = np.zeros([b, b], dtype=float)
    N = np.zeros([b, b], dtype=float)
    Us = np.zeros([b], dtype=float)
    for i in range(b):
        # print(cir_el_r[i])
        if 'r' == cir_el_r[i][0].lower():
            M[i, i] = 1
            N[i, i] = -cir_val_r[i, 0]
        elif 'i' == cir_el_r[i][0].lower():
            N[i, i] = 1
            Us[i] = cir_val_r[i, 0]
        elif 'v' == cir_el_r[i][0].lower():
            M[i, i] = 1
            Us[i] = cir_val_r[i, 0]
        elif 'a' == cir_el_r[i][0].lower():
            if 'in' in cir_el_r[i].lower():
                N[i, i] = 1
            elif 'ou' in cir_el_r[i].lower():
                M[i, i-1] = 1
            Us[i] = 0
        elif 'g' == cir_el_r[i][0].lower():
            for j in range(b):
                if cir_el_r[j].lower() == cir_ctrl_r[i].lower():
                    k = j
            M[i, k] = -cir_val_r[i, 0]
            N[i, i] = 1
            Us[i] = 0
        elif 'e' == cir_el_r[i][0].lower():
            for j in range(b):
                if cir_el_r[j].lower() == cir_ctrl_r[i].lower():
                    k = j
            M[i, k] = -cir_val_r[i, 0]
            M[i, i] = 1
            Us[i] = 0
        elif 'h' == cir_el_r[i][0].lower():
            for j in range(b):
                if cir_el_r[j].lower() == cir_ctrl_r[i].lower():
                    k = j
            M[i, i] = 1
            N[i, k] = -cir_val_r[i, 0]
            Us[i] = 0
        elif 'f' == cir_el_r[i][0].lower():
            for j in range(b):
                if cir_el_r[j].lower() == cir_ctrl_r[i].lower():
                    k = j
            N[i, k] = -cir_val_r[i, 0]
            N[i, i] = 1
            Us[i] = 0
        elif 'b' == cir_el_r[i][0].lower():
            M[i, i] = 1
            if t is None:
                Us[i] = cir_val_r[i, 0]
            else:
                Us[i] = cir_val_r[i, 0]*math.sin(2*math.pi*cir_val_r[i, 1]*t
                                                 + math.pi/180*cir_val_r[i, 2])
        elif 'y' == cir_el_r[i][0].lower():
            N[i, i] = 1
            if t is None:
                Us[i] = cir_val_r[i, 0]
            else:
                Us[i] = cir_val_r[i, 0]*math.sin(2*math.pi*cir_val_r[i, 1]*t
                                                 + math.pi/180*cir_val_r[i, 2])
        elif ('q' == cir_el_r[i][0].lower() and
              'be' == cir_el_r[i][-2:].lower()):
            g11, g12, g21, g22, Ie, Ic = zl3.trans_NR(cir_val_r[i][0],
                                                      cir_val_r[i][1],
                                                      cir_val_r[i][2],
                                                      Vj[i], Vj[i+1])
            M[i, i] = g11
            M[i, i+1] = g12
            N[i, i] = 1
            Us[i] = Ie
        elif ('q' == cir_el_r[i][0].lower() and
              'bc' == cir_el_r[i][-2:].lower()):
            g11, g12, g21, g22, Ie, Ic = zl3.trans_NR(cir_val_r[i][0],
                                                      cir_val_r[i][1],
                                                      cir_val_r[i][2],
                                                      Vj[i-1], Vj[i])
            M[i, i] = g22
            M[i, i-1] = g21
            N[i, i] = 1
            Us[i] = Ic
        elif 'd' == cir_el_r[i][0].lower():
            gd, Id = zl3.diode_NR(cir_val_r[i][0], cir_val_r[i][1], Vj[i])
            M[i, i] = gd
            N[i, i] = 1
            Us[i] = Id
        elif 'c' == cir_el_r[i][0].lower():
            M[i, i] = 1
            N[i, i] = -h/cir_val_r[i][0]
            Us[i] = cir_val_r[i][1]
        elif 'l' == cir_el_r[i][0].lower():
            M[i, i] = h/cir_val_r[i][0]
            N[i, i] = -1
            Us[i] = -cir_val_r[i][1]


    # print('M = ')
    # print(M)
    # print('N = ')
    # print(N)
    # print('Us = ')
    # print(Us)
    # print('A = ')
    # print(A)

    A_t = np.transpose(A)
    # print('A_t = ')
    # print(A_t)
    var_mat = np.empty([0], dtype=str)
    for i in range(n):
        var_mat = np.append(var_mat, ['e'+str(i+1)], axis=0)
    for i in range(b):
        var_mat = np.append(var_mat, ['v'+str(i+1)], axis=0)
    for i in range(b):
        var_mat = np.append(var_mat, ['i'+str(i+1)], axis=0)
    # print('var_mat = ')
    # print(var_mat)
    sol_mat = np.empty([0], dtype=float)
    for i in range(2*b):
        sol_mat = np.append(sol_mat, [0], axis=0)
    sol_mat = np.append(sol_mat, Us, axis=0)
    # print('sol_mat = ')
    # print(sol_mat)
    # At = np.transpose(A)

    Zero_1 = np.zeros([n, n], dtype=float)
    Zero_2 = np.zeros([n, b], dtype=float)
    Zero_3 = np.zeros([b, b], dtype=float)
    Zero_4 = np.zeros([b, n], dtype=float)
    Bat_m = np.eye(b)
    Tableau = np.empty([n+2*b, n+2*b], dtype=float)
    Tableau_0 = np.empty([n, n+2*b], dtype=float)
    for j in range(n):
        C = np.append(Zero_1[j], Zero_2[j], axis=0)
        D = np.append(C, A[j], axis=0)
        Tableau_0[j] = D
        # print(Tableau_0)
    Tableau_1 = np.empty([b, n+2*b], dtype=float)
    for j in range(b):
        E = np.append(-A_t[j], Bat_m[j], axis=0)
        F = np.append(E, Zero_3[j], axis=0)
        # print(E, F)
        Tableau_1[j] = F
        # print(Tableau_1)
    Tableau_2 = np.empty([b, n+2*b], dtype=float)
    for j in range(b):
        G = np.append(Zero_4[j], M[j], axis=0)
        H = np.append(G, N[j], axis=0)
        Tableau_2[j] = H
        # print(Tableau_2)
    for j in range(n):
        Tableau[j] = Tableau_0[j]
    for j in range(n, n+b):
        Tableau[j] = Tableau_1[j-n]
    for j in range(n+b, n+2*b):
        Tableau[j] = Tableau_2[j-n-b]
    soluzio = np.empty([n+2*b, 1], dtype=float)
    for j in range(n+2*b):
        if j < n+b:
            soluzio[j] = 0
        else:
            soluzio[j] = Us[j-n-b]
    # print(soluzio)
    try:
        emaitza_m = np.linalg.solve(Tableau, soluzio)
    except np.linalg.LinAlgError:
        sys.exit("Error solving Tableau equations, check if det(T) != 0.")
    return emaitza_m


def dc_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, A_r,
                n, b, f, filename):
    '''
    

    Parameters
    ----------
    cir_el_r : TYPE
        DESCRIPTION.
    cir_nd_r : TYPE
        DESCRIPTION.
    cir_val_r : TYPE
        DESCRIPTION.
    cir_ctrl_r : TYPE
        DESCRIPTION.
    A_r : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    start = float(f[5])
    finish = float(f[6])
    step = float(f[7])
    source = f[8]
    for i in range(len(cir_el_r)):
        if cir_el_r[i].lower() == source.lower():
            source_b = i  # branch of the source
            break

    filename = filename[:-4] + '_' + source + ".dc"

    if source[0].lower() == 'v' or source[0].lower() == 'b':
        header = zl2.build_csv_header('v', b, n)
    elif source[0].lower() == 'i' or source[0].lower() == 'y':
        header = zl2.build_csv_header('i', b, n)

    with open(filename, 'w') as file:
        print(header, file=file)
        for val in np.arange(start, finish+step, step):
            cir_val_r[source_b][0] = val
            if lineal:
                Vj = np.array([0])
                sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                                  b, A_r, n, Vj, step)
            else:
                sol = zl3.NR(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                             b, n, A_r, nodes, el_num, not_lin_elem)
            sol_csv = ','.join(['%.9f' % num for num in sol])
            sol_csv = str(val) + ', ' + sol_csv
            print(sol_csv, file=file)


def tr_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num, A_r,
                n, b, f, lineal, not_lin_elem, filename):
    '''
    This function gnerates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file.
    Parameters
    ----------
    cir_el_r : reshaped cir_el.
    cir_nd_r : reshaped cir_nd.
    cir_val_r : reshaped cir_val.
    cir_ctrl_r : reshaped cir_ctrl.
    b : number of branches.
    A : incidence matrix.
    n : number of nodes.
    f : TYPE
        DESCRIPTION.
    lineal : boolean. True if the circuit is lineal.
    filename : string with the filename (incluiding the path)
    '''

    # Sup .tr
    start = float(f[5])
    finish = float(f[6])
    step = float(f[7])
    filename = filename[:-3] + "tr"

    header = zl2.build_csv_header("t", b, n)
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        for t in np.arange(start, finish+step, step):
            # for t in tr["start"],tr["end"],tr["step"]
            # Recalculate the Us for the sinusoidal sources
            if lineal:
                Vj = np.array([0])
                sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                                  b, A_r, n, Vj, step, t)
            else:
                sol = zl3.NR(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                             b, n, A_r, nodes, el_num, not_lin_elem, step, t)
            if dynamic:
                for elem in dyn_elem:
                    if 'c' == cir_el_r[elem][0].lower():
                        cir_val_r[elem][1] = sol[n+elem-1]
                    if 'l' == cir_el_r[elem][0].lower():
                        cir_val_r[elem][1] = sol[n+b+elem-1]
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)


class NoReferenceNode(Exception):
    'Raised when there is no reference node.'
    pass


if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/0_zlel_serial_I_II.cir"

    [cir_el, cir_nd, cir_val, cir_ctrl, analisis] = zl2.cir_parser(filename)
    [cir_el_r, cir_nd_r, cir_val_r,
     cir_ctrl_r, b, n, nodes, el_num, lineal,
     not_lin_elem, dynamic, dyn_elem] = cir_reshape(
         cir_el, cir_nd, cir_val, cir_ctrl)
    Aa = zl1.incidence_matrix(cir_nd_r, b, n, nodes)
    zl1.check_inc_mat(cir_el, cir_nd, cir_val, cir_ctrl, Aa)
    sol = do_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num,
                      analisis, Aa, n, b, nodes, lineal, not_lin_elem,
                      filename)
