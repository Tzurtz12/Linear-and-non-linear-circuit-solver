#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Jon Gaminde & Urtzi Jauregi


"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

if __name__ == "zlel.zlel_p2":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


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

    cir_nd = np.array(cir[:, [1, 2, 3, 4]], dtype=int)
    if 0 not in cir_nd:
        raise NoReferenceNode()

    cir_val = np.array(cir[:, [5, 6, 7]], dtype=float)

    cir_ctrl = np.array(cir[:, [8]], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctrl, analisis


def do_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num, analisis,
                A, n, b, nodes, filename):
    '''
    

    Parameters
    ----------
    cir_el_r : reshaped cir_el.
    cir_nd_r : reshaped cir_nd.
    cir_val_r : reshaped cir_val.
    cir_ctrl_r : reshaped cir_ctrl.
    el_num : TYPE
        DESCRIPTION.
    analisis : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    nodes : TYPE
        DESCRIPTION.
    filename : input circuit.

    Returns
    -------
    None.

    '''
    for f in analisis:
        if f[0].lower() == '.pr':
            zl1.print_cir_info(cir_el_r, cir_nd_r, b, n, nodes, el_num)
            # zl1.print_incidence_matrix(cir_nd_r, b, n, nodes)
            print(A)
        elif f[0].lower() == '.op':
            sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                              b, A, n, 0)
            print_solution(sol, b, n)
        elif f[0].lower() == '.dc':
            dc_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                        A, n, b, f, filename)
            source = f[8]
            filename = filename[:-4] + '_' + source + ".dc"
            # save_as_csv(b, n, filename)
            plot_from_cvs(filename, "v", "e1", "")
        elif f[0].lower() == '.tr':
            tr_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                        A, n, b, f, filename)
            filename = filename[:-3] + "tr"
            # save_as_csv(b, n, filename)
            plot_from_cvs(filename, "t", "e1", "")


def elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, A, n, h,
                t=None):
    '''
    This is the key function of the whole program. Here, we create the LOKESEA
    matrices by taking into account the equations of the elements we might have
    in the circuits. Let's see the equations of the elements needed for this
    second task:
        Resistor --> math:'v-Ri=0'
        Current --> math:'i=I'
        Voltage --> math:'v=V'
        Amplifier --> math:''
        VCCS --> math:''
        VCVS --> math:''
        CCVS --> math:''
        CCCS --> math:''
        Sinusoidal VS --> math:''
        Sinusoidal CS --> math:''
    Parameters
    ----------
    cir_el_r : reshaped cir_el.
    cir_nd_r : reshaped cir_nd.
    cir_val_r : reshaped cir_val.
    cir_ctrl_r : reshaped cir_ctrl.
    b : number of branches.
    A : incidence matrix.
    n : number of nodes.
    h : step for NR.
    t : time variable for sinusoidal souces.

    Returns
    -------
    The function creates M, N and U Tableau matrices and returns the lineal
    solution.

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
    # Tableau = np.array([Tableau_0,Tableau_1,Tableau_2],dtype=object)
    # print(Tableau)
    # tableau_mat = np.zeros([n+2*b, n+2*b])
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
    # print(var_mat)
    # print(emaitza_m)
    return emaitza_m


def dc_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, A,
                n, b, f, filename):
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
        header = build_csv_header('v', b, n)
    elif source[0].lower() == 'i' or source[0].lower() == 'y':
        header = build_csv_header('i', b, n)

    with open(filename, 'w') as file:
        print(header, file=file)
        for val in np.arange(start, finish+step, step):
            cir_val_r[source_b][0] = val
            sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                              b, A, n, 0)
            sol_csv = ','.join(['%.9f' % num for num in sol])
            sol_csv = str(val) + ', ' + sol_csv
            print(sol_csv, file=file)


def tr_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, A,
                n, b, f, filename):
    """ This function generates a csv file with the name filename.
        First it will save a header and then, it loops and save a line in
        csv format into the file.

    Args:
        b: # of branches
        n: # of nodes
        filename: string with the filename (incluiding the path)
    """
    # Sup .tr
    start = float(f[5])
    finish = float(f[6])
    step = float(f[7])

    filename = filename[:-3] + "tr"

    header = build_csv_header("t", b, n)
    with open(filename, 'w') as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        for t in np.arange(start, finish+step, step):
            # for t in tr["start"],tr["end"],tr["step"]
            # Recalculate the Us for the sinusoidal sources
            sol = elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
                              b, A, n, step, t)
            # Inserte the time
            sol = np.insert(sol, 0, t)
            # sol to csv
            sol_csv = ','.join(['%.9f' % num for num in sol])
            print(sol_csv, file=file)


def print_solution(sol, b, n):
    """ This function prints the solution with format.

        Args:
            sol: np array with the solution of the Tableau equations
            (e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b)
            b: # of branches
            n: # of nodes

    """

    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=' ')  # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind] = np.array(sol[ind])
        sol = tmp
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        print("e" + str(i) + " = ", sol[i-1])
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        print("v" + str(i) + " = ", sol[i+n-2])
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        print("i" + str(i) + " = ", sol[i+b+n-2])

    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """ This function build the csv header for the output files.
        First column will be v or i if .dc analysis or t if .tr and it will
        be given by argument tvi.
        The header will be this form,
        t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b

    Args:
        tvi: "v" or "i" if .dc analysis or "t" if .tran
        b: # of branches
        n: # of nodes

    Returns:
        header: The header in csv format as string
    """
    header = tvi
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b+1):
        header += ",v" + str(i)
    for i in range(1, b+1):
        header += ",i" + str(i)
    return header


def plot_from_cvs(filename, x, y, title):
    """ This function plots the values corresponding to the x string of the
        file filename in the x-axis and the ones corresponding to the y
        string in the y-axis.
        The x and y strings must mach with some value of the header in the
        csv file filename.

    Args:
        filename: string with the name of the file (including the path).
        x: string with some value of the header of the file.
        y: string with some value of the header of the file.

    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=0,
                         skip_footer=1, names=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data[x], data[y], color='r', label=title)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    plt.show()


class NotCorrectFile(Exception):
    'Raised when the structure of the data in the file is not correct.'
    pass


class NoReferenceNode(Exception):
    'Raised when there is no reference node.'
    pass


class SingleConexionNode(Exception):
    'Raised when a node has just one conexion.'
    pass


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_ekorketa.cir"

    [cir_el, cir_nd, cir_val, cir_ctrl, analisis] = cir_parser(filename)
    [cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r,
     b, n, nodes, el_num] = zl1.cir_reshape(cir_el, cir_nd,
                                            cir_val, cir_ctrl)
    A = zl1.incidence_matrix(cir_nd_r, b, n, nodes)
    A_r = zl1.reduce_incidence_matrix(A)
    sol = do_analisis(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, el_num,
                      analisis, A_r, n, b, nodes, filename)
    # elem_matrix(cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, A_r, n)

    # b = 2
    # n = 2
