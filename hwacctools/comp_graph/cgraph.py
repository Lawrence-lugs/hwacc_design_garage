import onnx
from onnx import numpy_helper as nphelp
import numpy as np
from . import cnodes
from tqdm import tqdm
from . import splitter
from . import cnode_factory
import os
from .. import onnx_utils

class Cgraph(object):
    '''
    A computational graph

    Attributes
    ----------
    nodes : list[cnodes.node]
        A list of computational nodes which perform functions on
        the arrays from its input edges as passes 
    edges : dict[str,np.ndarray]
        A dictionary of tensors which serve as inputs and outputs
        of nodes on the cgraph.
    '''

    def __init__(self,node_list:list[cnodes.Node],cachePath=None): 
        '''
        Parameters
        ---------
        nodes : list[cnodes.node]
            List of computational nodes from which to build the cgraph 
        '''

        self.nodes = node_list
        
        self.edges = dict()
        for node in self.nodes:
            for input in node.inputs:
                self.edges[input] = None
            for output in node.outputs:
                self.edges[output] = None
        self.cachePath = cachePath

    def get_cgraph_node_by_output(self,output):
        '''
        Returns the node corresponding to the output
        '''
        for node in self.nodes:
            if output in node.outputs:
                return node
        raise ValueError(f'Output {output} not found in cgraph')

    @classmethod
    def from_onnx_model(cls,nx_model, tiles=None, channel_minor=False, **kwargs):
        '''
        Obtains a `cgraph` from an ONNX model loaded in.
        '''
        node_list = []
        for node in nx_model.graph.node:
            a = cnode_factory.get_cnode_from_onnx_node(node, nx_model, channel_minor=True)
            if type(a) == list:
                node_list.extend(a)
            else:
                node_list.append(a)
        return Cgraph(node_list,**kwargs)

    def check_if_node_done(self,node):
        '''
        Checks if the outputs of a node is already filled
        '''
        for output in node.outputs:
            if self.edges[output] is None:
                return False
        return True

    def check_if_node_ready(self,node):
        '''
        Checks if the inputs of a node is already filled
        '''
        for input in node.inputs:
            if self.edges[input] is None:
                return False
        return True
    
    def ready_nodes(self):
        '''
        Returns a list of nodes whose input edges are ready 
        '''
        node_list = []
        for node in self.nodes:
            if self.check_if_node_done(node):
                continue
            if self.check_if_node_ready(node):
                node_list.append(node)
        return node_list
    
    def single_node_forward(self,node):
        '''
        Runs a single node's forward pass
        '''
        in_array = []
        for input in node.inputs:
            in_array.append(self.edges[input])

        if self.cachePath is not None:
            try:
                out_array = np.load(self.cachePath + f'/{node.outputs[0]}.npy',allow_pickle=True)
            except FileNotFoundError:
                out_array = node.forward(in_array)
                np.save(self.cachePath + f'/{node.outputs[0]}.npy',out_array) 
        else:
            out_array = node.forward(in_array)

        for output in node.outputs:
            self.edges[output] = out_array

    def forward(self,input_dict:dict,output_keys:list[str] = None,verbose=False, recalculate=False, progbar=True):
        '''
        Parameters
        ----------
        input_dict : dictionary
            dictionary that contains keys corresponding to keys
            in the edges dictionary, and values corresponding to numpy
            array inputs to place in those edges.
        output_keys : list[str], optional
            list containing the keys of the edges whose values will
            be returned after inference.
        verbose : bool, optional
            If True, prints the output of each node.
        self.cachePath : str, optional
            If specified, the output of each node will be saved in
            the self.cachePath directory.
        recalculate : bool, optional
            If true, the output of each node will be recalculated regardless of edge existence in self.cachePath
        progbar : bool, optional
            If True, displays a progress bar for the forward pass.
        '''

        if self.cachePath is not None:
            if not os.path.exists(self.cachePath):
                os.makedirs(self.cachePath)
        
        for key in input_dict:
            self.edges[key] = input_dict[key]

        while not self.check_if_node_done(self.nodes[-1]):
            for node in tqdm(self.nodes,disable=not progbar):
                if not self.check_if_node_ready(node):
                    continue
                if self.check_if_node_done(node):
                    continue
                if verbose: print(f'node:{node},\n output: {node.outputs}')
                self.single_node_forward(node)

            if output_keys is None:
                # Return the output of the last node
                output_keys = self.nodes[-1].outputs

            return [self.edges[x] for x in output_keys]
    
    def _get_shape_list_id(self, excludeDepthwise=True):
        '''
        Obtains the list of shapes of all matrix-containing nodes
        and assigns an ID to each depending on their position in the node list
        '''
        shapes = []
        for id,node in enumerate(self.nodes):
            if hasattr(node,"matrix"):
                # we remove the skip functionality for depthwise convolutions because dwc_nodes node longer have a matrix attribute
                shapes.append( (*node.matrix.shape,id) )
            node.rid = id
        return shapes

    def color_by_separable_type(self):
        '''
        Colors pointwise, depthwise, gemms, and conv nodes.

        red = gemm
        blue = pointwise conv
        yellow = regular conv

        Depthwise convolutions are not included in the coloring.
        '''

        for node in self.nodes:
            if hasattr(node,'from_type'):
                if node.from_type == 'gemm':
                    node.color = '#f1948a' # Red
                elif node.from_type == 'pointwise':
                    node.color = '#85c1e9' # Blue
                elif node.from_type == 'conv':
                    node.color = '#58d68d' # Yellow
                else:
                    node.color = 'white'

        return self.generate_color_dict()
    
    def color_by_nid(self):

        highnumber = 12312941249
        for node in self.nodes:
            rand = highnumber*(node.nid)
            node.color = f'#{hex(rand)[2:8]}'
        return self.generate_color_dict()

    def generate_color_dict(self):
        '''
        Generates a color dictionary for the nodes in the cgraph
        '''
        color_dict = dict()
        for node in self.nodes:
            if hasattr(node,'color') and hasattr(node,'rid'):
                color_dict[node.rid] = node.color
        return color_dict
    
    def rid_to_nid_dict(self):
        '''
        Generates a dictionary that maps rid to nid
        '''
        rid_to_nid = dict()
        for node in self.nodes:
            if hasattr(node,'rid'):
                rid_to_nid[node.rid] = node.nid
        return rid_to_nid

    def get_matrix_dict(self):
        '''
        Returns a dictionary mapping node index to matrix for all
        matrix-containing nodes in the cgraph.
        '''
        mx_dict = dict()
        for i, node in enumerate(self.nodes):
            if hasattr(node, 'matrix'):
                mx_dict[i] = node.matrix
        return mx_dict


def split_convolutions(in_cgraph:Cgraph,H:int,W:int):
    '''
    Limits the cgraph's matrices to a size limit of H, W.
    If the matrix's size is larger, the matrix is split and 
    additional nodes to manage this computation is added.
    '''
    new_nodes = []

    for nid,node in enumerate(in_cgraph.nodes):
        if type(node) == cnodes.conv_node:
            replacements = splitter.split_conv_into_chunks(node,H,W)
            for node in replacements:
                node.nid = nid
            new_nodes.extend(replacements)
        elif type(node) == cnodes.gemm_node:
            replacements = splitter.split_gemm_into_chunks(node,H,W)
            for node in replacements:
                node.nid = nid
            new_nodes.extend(replacements)
        else:
            node.nid = nid
            new_nodes.append(node)

    return Cgraph(new_nodes)

def compare_with_onnx(
    modelpath,
    cgraph,
    input_tensor_name,
    output_tensor_name,
    cgraph_input_dict,
    verbose=False
):

    tensor_in = onnx_utils.get_intermediate_tensor_value(
        modelpath, 
        tensor_name= input_tensor_name,
        input_dict = cgraph_input_dict
    )

    scaler_node = cgraph.get_cgraph_node_by_output(output_tensor_name)
    conv_node = cgraph.get_cgraph_node_by_output(input_tensor_name + '_scaler_input')
    interm = conv_node.forward([tensor_in])
    cgraph = scaler_node.forward([interm])

    # input_name = onnx.load(modelpath).graph.input[0].name

    ref = onnx_utils.get_intermediate_tensor_value(
        modelpath, 
        tensor_name=output_tensor_name,
        input_dict= cgraph_input_dict
    )

    # if (verbose) : # Print the cgraph and ref tensors
    #     print(f'cgraph shape: {cgraph.shape}')
    #     print(f'ref shape: {ref.shape}')
    #     print(f'cgraph: {cgraph}')
    #     print(f'ref: {ref}')

    if (cgraph == ref).all():
        print('PASSED')
    #     return scaler_node, conv_node, cgraph, ref, interm
    else:
        # print(scaler_node, conv_node)
        # print(scaler_node.outputs, conv_node.outputs)
        # print(f'shape: {cgraph.shape}')
        # print(f'pre-scaling: {interm}')
        print(f'maxdiff: {(cgraph - ref).max()}')
        print(f'mindiff: {(cgraph - ref).min()}')
        # print(f'onnx_output: {ref}')
        print('FAILED')
        # return scaler_node, conv_node, cgraph, ref, interm

    return np.allclose(cgraph,ref,atol=2) # True