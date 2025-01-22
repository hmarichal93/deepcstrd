from cross_section_tree_ring_detection.sampling import sampling_edges as sampling_edges_cstrd
import cross_section_tree_ring_detection.chain as ch



def remove_duplicated_elements(l_ch_s, l_nodes_s):
    """
    This control must be done becuase the find contour method of opencv can return the same contour multiple times or with a high degree of overlapping.
    :param l_ch_s:
    :return:
    """
    l_ch_s_aux = []
    #sorted by size l_ch_s
    l_ch_s = sorted(l_ch_s, key=lambda x: len(x.l_nodes), reverse=True)
    for chain_1 in l_ch_s:
        nodos_in_common = False
        for chain_2 in l_ch_s_aux:
            if chain_1 == chain_2:
                if chain_1.id != chain_2.id:
                    #nodes in common but different id. Chain that we want to delete
                    nodos_in_common = True

                continue
            #check if they have nodes in common
            for node in chain_1.l_nodes:
                if node in chain_2.l_nodes:
                    nodos_in_common = True
                    break

        if not nodos_in_common:
            l_ch_s_aux.append(chain_1)




    if len(l_ch_s_aux) != len(l_ch_s):
        print(f"Removed {len(l_ch_s) - len(l_ch_s_aux)} duplicated chains")
        l_ch_s = l_ch_s_aux
        #chaing the chain id
        for i, chain in enumerate(l_ch_s):
            chain.change_id(i)
            chain.label_id = chain.id
        l_nodes_s = []
        for chain in l_ch_s:
            l_nodes_s += chain.l_nodes



    return l_ch_s, l_nodes_s

def control_check_duplicated_chains_in_list(l_within_chains):
    """
    Check duplicated chains in list
    @param l_within_chains: list of chains
    @return:
    """
    l_ch_s_aux = []
    for chain in l_within_chains:
        if chain not in l_ch_s_aux:
            l_ch_s_aux.append(chain)
    if len(l_ch_s_aux) != len(l_within_chains):
        raise Exception("Duplicated chains in list")

def control_check_spyder_web_hypotesis(l_ch_s, l_nodes_s, debug_output_dir):
    """
    Check if the spyder web hypotesis is satisfied. There is no nodes duplicated in the list. A node is formed by
    an x coordinate, y coordinate and an angle.
    :param l_ch_s:
    :param l_nodes_s:
    :param debug_output_dir:
    :return:
    """
    l_nodes_s_aux = []
    for node in l_nodes_s:
        if node not in l_nodes_s_aux:
            l_nodes_s_aux.append(node)
    if len(l_nodes_s_aux) != len(l_nodes_s):
        raise Exception("Spyder web hypotesis not satisfied")

    #check if the chains are correct
    for chain in l_ch_s:
        for node in chain.l_nodes:
            if node not in l_nodes_s:
                raise Exception("Spyder web hypotesis not satisfied")
    print("Spyder web hypotesis satisfied")

    return



def sampling_edges(l_ch_f, cy, cx, im_pre, mc, nr, debug=False, debug_output_dir=None):
    """

    @param l_ch_f:  edges devernay curves
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param im_pre: input image
    @param nr: total ray number
    @param mc:  minumim chain length
    @param debug: debugging flag
    @return:
    - l_ch_s: sampled edges curves. List of chain objects
    - l_nodes_s: nodes list.
    """
    l_ch_s, l_nodes_s = sampling_edges_cstrd(l_ch_f, cy, cx, im_pre, mc, nr, debug)

    #return im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, [], [], []
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            l_ch_s, [], img=im_pre, filename=f'{debug_output_dir}/chains_origin.png')

    l_ch_s, l_nodes_s = remove_duplicated_elements(l_ch_s, l_nodes_s)

    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(
            l_ch_s, [], img=im_pre, filename=f'{debug_output_dir}/chains_with_no_duplication.png')

    control_check_spyder_web_hypotesis(l_ch_s, l_nodes_s, debug_output_dir)
    # Line 5
    return l_ch_s, l_nodes_s