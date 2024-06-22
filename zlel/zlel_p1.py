#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Jon Gaminde & Urtzi Jauregi


"""

import numpy as np
import sys


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
        NotCorrectFile

    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")

    for lerro in cir:
        if len(lerro) != 9:
            raise NotCorrectFile('The structure of the data in the file is '
                                 'not correct.')

    cir_el = np.array(cir[:, 0], dtype=str)

    cir_nd = np.array(cir[:, [1, 2, 3, 4]], dtype=int)

    cir_val = np.array(cir[:, [5, 6, 7]], dtype=float)

    cir_ctrl = np.array(cir[:, [8]], dtype=str)

    return cir_el, cir_nd, cir_val, cir_ctrl


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
        In order to get the reshaped matrices and number of branches, we need
        to realize that there are some elements connecting several branches as
        the amplifier or transistors. To recognise those elements we take into
        account their identification from the input circuit. The amplifiers
        Axx and transistors Qxx. With this function we extend the rows of the
        matrices and get the correct number of branches easily.

    """

    el_num = len(cir_el)

    cir_el_r = np.empty([0], dtype=str)
    cir_nd_r = np.empty([0, 2], dtype=int)
    cir_val_r = np.empty([0, 3], dtype=float)
    cir_ctrl_r = np.empty([0], dtype=str)

    for i in range(el_num):
        if 'A' in cir_el[i]:
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
        elif 'Q' in cir_el[i]:
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
    nodes = np.unique(cir_nd_r)
    if 0 not in nodes:
        raise NoReferenceNode('Reference node "0" is not defined in the '
                              'circuit.')
    n = len(nodes)
    return cir_el_r, cir_nd_r, cir_val_r, cir_ctrl_r, b, n, nodes, el_num


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """ Prints the info of the circuit:
            1.- Elements info
            2.- Node info
            3.- Branch info
            4.- Variable info
    Args:
        cir_el: reshaped cir_el
        cir_nd: reshaped cir_nd. Now it will be a (b,2) matrix
        b: # of branches
        n: # number of nodes
        nodes: an array with the circuit nodes sorted
        el_num:  the # of elements.

    """
    # Element info
    print(str(el_num) + ' Elements')
    # Node info
    print(str(n) + ' Different nodes: ' +
          str(nodes))
    # Branch info
    print("\n" + str(b) + " Branches: ")

    for i in range(1, b+1):
        print("\t" + str(i) + ". branch:\t" + cir_el[i-1] +
              ",\ti" + str(i) +
              ",\tv" + str(i) +
              "=e" + str(cir_nd[i-1, 0]) +
              "-e" + str(cir_nd[i-1, 1]))

    # Variable info
    print("\n" + str(2*b + (n-1)) + " variables: ")
    # Print all the nodes but the first (0 because is sorted)
    for i in nodes[1:]:
        print("e"+str(i)+", ", end="", flush=True)
    for i in range(b):
        print("i"+str(i+1)+", ", end="", flush=True)
    # Print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b-1):
        print("v"+str(i+1)+", ", end="", flush=True)
    print("v"+str(b))

    # IT IS RECOMMENDED TO USE THIS FUNCTION WITH NO MODIFICATION.


def incidence_matrix(cir_nd_r, b, n, nodes):
    """
        This function creates the Incidence Matrix of a circuit.

    Args:
        cir_nd_r: reshaped cir_nd
        b: # of branches
        n: # of nodes
        nodes: an array with the circuit nodes sorted

    Returns:
        Aa: incidence matrix of the circuit

    Rises:
        SystemExit

    """
    Aa = np.zeros([n, b])
    for i in range(b):
        for j in range(2):
            nd_pos = np.where(nodes == cir_nd_r[i, j])
            # print(nd_pos)
            if j == 0:
                Aa[nd_pos, i] = 1
            else:
                Aa[nd_pos, i] = -1
    # print(Aa)
    for n in range(len(Aa)):
        val = 0
        for b in range(len(Aa[n])):
            val += abs(Aa[n][b])
        if val < 2:
            raise SingleConexionNode('Node ' + str(n) + ' is floating.')
    return Aa


def check_inc_mat(cir_el, cir_nd, cir_val, cir_ctrl, Aa):
    v_source_num = 0
    c_source_num = 0
    for el_i in range(len(cir_el)):
        if 'v' == cir_el[el_i][0].lower() or 'b' == cir_el[el_i][0].lower():
            v_source_num += 1
        if 'i' == cir_el[el_i][0].lower() or 'y' == cir_el[el_i][0].lower():
            c_source_num += 1
    if v_source_num > 1:
        parallel_v_source(cir_el, cir_nd, cir_val)
    elif c_source_num > 1:
        series_c_source(cir_el, cir_val, Aa)


def parallel_v_source(cir_el, cir_nd, cir_val):
    """
    We will use this function to ensure there are no parallel voltage sources.
    ----------
    Parameters
    ----------
    cir_el : np array that contains the elements of the circuit.
    cir_nd : np array containing the nodes of the circuir.
    cir_val : np array with the values of each element.

    Raises
    ------
    ParallelVSources
    If indeed, there are parallel sources in the defined circuit, an error
    string will raise noticing us of the problem. For that, we simply check if
    there are more than one source in the circuit and then see whether they are
    parallel to each other by looking to their nodes. Take care that we not
    only need to consider 'v' DC sources but also 'b' sinusoidal AC sources!

    Returns
    -------
    Location of the parallel sources
    """
    for el_i in range(len(cir_el)):
        if 'v' == cir_el[el_i][0].lower() or 'b' == cir_el[el_i][0].lower():
            for el_j in range(len(cir_el)):
                if el_i != el_j and ('v' == cir_el[el_j][0].lower() or
                                     'b' == cir_el[el_j][0].lower()):
                    if ((cir_nd[el_i][0] == cir_nd[el_j][0] and
                         cir_nd[el_i][1] == cir_nd[el_j][1]) and
                            (cir_val[el_i][0]) != (cir_val[el_j][0])):
                        raise ParallelVSources('Parallel V sources at '
                                               'branches ' + str(el_i) +
                                               ' and ' + str(el_j))
                    elif ((cir_nd[el_i][0] == cir_nd[el_j][1] and
                           cir_nd[el_i][1] == cir_nd[el_j][0]) and
                            (cir_val[el_i][0]) != -(cir_val[el_j][0])):
                        raise ParallelVSources('Parallel V sources at '
                                               'branches ' + str(el_i) +
                                               ' and ' + str(el_j))


def series_c_source(cir_el, cir_val, Aa):
    """
    This function is used to check there are no serial current sources in the
    circuit.
    Parameters
    ----------
    cir_el : np array that contains the elements of the circuit.
    cir_val : np array with the values of each element.

    Aa : incidence matrix of the circuit

    Raises
    ------
    SeriesCSources
    An error string will raise in case there are series current sources. It is
    simple to check by using the incidence matrix as it tells the in and out
    currents from each node. This way, if the resultant for each column is not
    0, an error shoul raise as a there are different currents on same wires.

    Returns
    -------
    None.

    """
    for node in range(len(Aa)):
        kcl_sum = 0
        for branch in range(len(Aa[0])):
            if Aa[node][branch] != 0:
                if ('i' != cir_el[branch][0].lower() or
                        'y' != cir_el[branch][0].lower()):
                    kcl_sum = 0
                    break
                else:
                    kcl_sum += Aa[node][branch]*cir_val[branch][0]
        if kcl_sum != 0:
            raise SeriesCSources('I sources in series at node ' + str(node))


def reduce_incidence_matrix(A):
    """

    Parameters
    ----------
    A : incidence matrix of the circuit.

    Returns
    -------
    A_r : reduced incidende matrix. We just delete the reference row of the
    original incidence matrix.

    """
    A_r = np.delete(A, 0, 0)
    return A_r


class NotCorrectFile(Exception):
    'Raised when the structure of the data in the file is not correct.'
    pass


class NoReferenceNode(Exception):
    'Raised when there is no reference node.'
    pass


class SingleConexionNode(Exception):
    'Raised when a node has just one conexion.'
    pass


class ParallelVSources(Exception):
    'Raised when voltage sources are in parallel and have different values.'
    pass


class SeriesCSources(Exception):
    'Raised when current sources are in series and have different values.'
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
        filename = "../cirs/all/0_zlel_node_float.cir"
    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctrl] = cir_parser(filename)
    # Reshape the matrix
    [cir_el_r, cir_nd_r, cir_val_r,
     cir_ctrl_r, b, n, nodes, el_num] = cir_reshape(cir_el, cir_nd,
                                                    cir_val, cir_ctrl)
    Aa = incidence_matrix(cir_nd_r, b, n, nodes)
    check_inc_mat(cir_el, cir_nd, cir_val, cir_ctrl, Aa)
    # Print all the info
    print_cir_info(cir_el_r, cir_nd_r, b, n, nodes, el_num)
    print('Incidence Matrix:')
    print(Aa)
