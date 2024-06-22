#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import numpy as np
import sys
import math

if __name__ == "zlel.zlel_p3":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matices.
        If the file has not the proper dimensions it warns and exit.

    Args:
        filename: string with the name of the file

    Returns:
        cir_el: np array of strings with the elements to parse. size(1,b)
        cir_nd: np array with the nodes to the circuit. size(b,4)
        cir_val: np array with the values of the elements. size(b,3)
        cir_ctrl: np array of strings with the element which branch
        controls the controlled sources. size(1,b)

    Rises:
        SystemExit

    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    for lerro in cir:
        if len(lerro) != 9:
            raise NotCorrectFile('The structure of the data in ',
                                 'the file is not correct.')

    analisis = np.empty([0, 9], dtype=str)
    el_num = len(cir)
    i = 0
    while i < (el_num):
        if '.' in cir[i, 0]:
            analisis = np.append(analisis, [cir[i]], axis=0)
            cir = np.delete(cir, i, 0)
            i -= 1
            el_num -= 1
        i += 1

    cir_el = np.array(cir[:, 0], dtype=str)
    # print(cir_el)
    cir_nd = np.array(cir[:, [1, 2, 3, 4]], dtype=int)
    if 0 not in cir_nd:
        raise NoReferenceNode()

    cir_val = np.array(cir[:, [5, 6, 7]], dtype=float)

    cir_ctrl = np.array(cir[:, [8]], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctrl, analisis


def cir_reshape(cir_el, cir_nd, cir_val, cir_ctrl):
    """
        This function takes the 4 matices from the circuit.

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

    for i in range(el_num):
        if 'a' in cir_el[i].lower():
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_in")
            in_nd = np.array([[cir_nd[i, 0], cir_nd[i, 1]]])
            cir_nd_r = np.append(cir_nd_r, in_nd, axis=0)
            val = np.array([[0, 0, 0]])
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, '0')
            cir_el_r = np.append(cir_el_r, cir_el[i]+"_out")
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
        elif 'd' in cir_el[i].lower():
            cir_el_r = np.append(cir_el_r, cir_el[i])
            nd = np.array([cir_nd[i][0:2]])
            cir_nd_r = np.append(cir_nd_r, nd, axis=0)
            val = np.array([cir_val[i]])
            cir_val_r = np.append(cir_val_r, val, axis=0)
            cir_ctrl_r = np.append(cir_ctrl_r, cir_ctrl[i])
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
    nodes = np.unique(cir_nd_r)
    # print(not_lin_elem)
    # print(cir_el_r)
    if 0 not in nodes:
        raise NoReferenceNode('Reference node "0" is not defined in the '
                              'circuit.')
    n = len(nodes)
    return (cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, n, nodes,
            el_num, lineal, not_lin_elem)


def elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, A, n,
                Vj, t=1):
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
            M[i, i] = 1
            N[i, i] = 1
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
            Us[i] = cir_val_r[i, 0]*math.sin(2*math.pi*cir_val_r[i, 1]*t
                                             + math.pi/180*cir_val_r[i, 2])
        elif 'y' == cir_el_r[i][0].lower():
            N[i, i] = 1
            Us[i] = cir_val_r[i, 0]*math.sin(2*math.pi*cir_val_r[i, 1]*t
                                             + math.pi/180*cir_val_r[i, 2])
        elif ('q' == cir_el_r[i][0].lower() and
              'be' == cir_el_r[i][-2:].lower()):
            g11, g12, g21, g22, Ie, Ic = trans_NR(cir_val_r[i][0],
                                                  cir_val_r[i][1],
                                                  cir_val_r[i][2],
                                                  Vj[i], Vj[i+1])
            M[i, i] = g11
            M[i, i+1] = g12
            N[i, i] = 1
            Us[i] = Ie
        elif ('q' == cir_el_r[i][0].lower() and
              'bc' == cir_el_r[i][-2:].lower()):
            g11, g12, g21, g22, Ie, Ic = trans_NR(cir_val_r[i][0],
                                                  cir_val_r[i][1],
                                                  cir_val_r[i][2],
                                                  Vj[i-1], Vj[i])
            M[i, i] = g22
            M[i, i-1] = g21
            N[i, i] = 1
            Us[i] = Ic
        elif 'd' == cir_el_r[i][0].lower():
            gd, Id = diode_NR(cir_val_r[i][0], cir_val_r[i][1], Vj[i])
            M[i, i] = gd
            N[i, i] = 1
            Us[i] = Id

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
        E = np.append(A_t[j], Bat_m[j], axis=0)
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
    # print(var_mat, emaitza_m)
    return emaitza_m


def diode_NR(I0, nd, Vdj):
    """ https://documentation.help/Sphinx/math.html
        Calculates the g and the I of a diode for a NR discrete equivalent
        Given,

        :math:`Id = I_0(e^{(\\frac{V_d}{nV_T})}-1)`

        The NR discrete equivalent will be,

        :math:`i_{j+1} + g v_{j+1} = I`

        where,

        :math:`g = -\\frac{I_0}{nV_T}e^{(\\frac{V_d}{nV_T})}`

        and

        :math:`I = I_0(e^{(\\frac{V_{dj}}{nV_T})}-1) + gV_{dj}`

    Args:
        I0: Value of I0.
        nD: Value of nD.
        Vd: Value of Vd.

    Return:
        gd: Conductance of the NR discrete equivalent for the diode.
        Id: Current independent source of the NR discrete equivalent.

    """

    Vt = 8.6173324e-5*300

    gd = -I0/nd/Vt*math.exp(Vdj/nd/Vt)

    Id = I0*(math.exp(Vdj/nd/Vt)-1)+gd*Vdj

    return gd, Id


def trans_NR(Ies, Ics, betaf, Vbej, Vbcj):
    """ https://documentation.help/Sphinx/math.html
        Calculates the g and the I of a diode for a NR discrete equivalent
        Given,

        :math:`Id = I_0(e^{(\\frac{V_d}{nV_T})}-1)`

        The NR discrete equivalent will be,

        :math:`i_{j+1} + g v_{j+1} = I`

        where,

        :math:`g = -\\frac{I_0}{nV_T}e^{(\\frac{V_d}{nV_T})}`

        and

        :math:`I = I_0(e^{(\\frac{V_{dj}}{nV_T})}-1) + gV_{dj}`

    Args:
        I0: Value of I0.
        nD: Value of nD.
        Vd: Value of Vd.

    Return:
        gd: Conductance of the NR discrete equivalent for the diode.
        Id: Current independent source of the NR discrete equivalent.

    """

    Vt = 8.6173324e-5*300

    alphaf = betaf/(1+betaf)
    alphar = Ies/Ics*alphaf

    g11 = -Ies/Vt*math.exp(Vbej/Vt)
    g22 = -Ics/Vt*math.exp(Vbcj/Vt)
    g12 = -alphar*g22
    g21 = -alphaf*g11

    Ie = (g11*Vbej+g12*Vbcj+Ies*(math.exp(Vbej/Vt) - 1) -
          alphar*Ics*(math.exp(Vbcj/Vt)-1))
    Ic = (g21*Vbej+g22*Vbcj-alphaf*Ies*(math.exp(Vbej/Vt)-1) +
          Ics*(math.exp(Vbcj/Vt)-1))
    # N1 = math.exp(Vbej/Vt)-1
    # N2 = -alphar*(math.exp(Vbcj/Vt)-1)
    # N3 = -alphaf*(math.exp(Vbej/Vt)-1)
    # N4 = math.exp(Vbcj/Vt)-1
    return g11, g12, g21, g22, Ie, Ic   # N1, N2, N3, N4


def NR(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, n, A_r, nodes, el_num,
       not_lin_elem):
    N = 100
    e0 = 1e-5
    print(not_lin_elem)
    Vj = np.zeros([b, 1], dtype=float)
    for i in not_lin_elem:
        Vj[i][0] = 0.6

    print(Vj)
    it = 0

    while True:
        sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, A_r,
                          n, Vj)
        Vj1 = sol[n-1:n+b-1]
        if it > N:
            print('iterazio')
            print(sol)
            break
        a = 0
        for j in range(len(Vj)):
            # print(abs(Vj1[j]-Vj[j]))
            if j in not_lin_elem and abs(Vj1[j]-Vj[j]) < e0:
                print('Errorea: '+str(abs(Vj1[j]-Vj[j])))
                a += 1
        if a == len(not_lin_elem):
            print('Iterazioa: '+str(it))
            print(sol)
            break
        else:
            # print(Vj)
            Vj = Vj1
            it += 1


class NotCorrectFile(Exception):
    'Raised when the structure of the data in the file is not correct.'
    pass


class NoReferenceNode(Exception):
    'Raised when there is no reference node.'
    pass


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_Q_ezaugarri.cir"
        # filename = "../cirs/all/2_zlel_Q.cir"

    [cir_el, cir_nd, cir_val, cir_ctrl, analisis] = cir_parser(filename)
    [cir_el_r, cir_nd_r, cir_val_r,
     cir_ctrl_r, b, n, nodes, el_num, lineal, not_lin_elem] = cir_reshape(
         cir_el, cir_nd, cir_val, cir_ctrl)
    Aa = zl1.incidence_matrix(cir_nd_r, b, n, nodes)
    A_r = zl1.reduce_incidence_matrix(Aa)
    if not lineal:
        sol = NR(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, n, A_r,
                 nodes, el_num, not_lin_elem)

#    end = time.perf_counter()
#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds
