#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:32:37 2019

@author: gbmello
"""

# Import Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
from skimage.filters import threshold_otsu
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import itertools
from skimage.morphology import skeletonize
import networkx as nx


###############################################################################
# Define functions
###############################################################################
def visualize_labeled_connected_components(labels, show =1):
    """
    Function maps the label of the Connected component to a Hue
    It creates an image and outputs it as a colorful image.
    INPUT:
        labels = (uint8 ndarray) labeled connected component image
    OUTPUT:
        labeled_img = (ndarray) m x n x 3 image array
    CREDIT:
        https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python?rq=1
    """
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
    # set bg label to black
    labeled_img[label_hue==0] = 0
    if show == 1:
        plt.imshow( labeled_img)
        plt.show()
    return labeled_img


# Functions to generate kernels of curve intersection 
def generate_nonadjacent_combination(input_list,take_n):
    """ 
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination 
        take_n =     (integer) Number of elements that you are going to take at a time in
                     each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        fd = np.diff(np.flip(comb))
        if len(d[d==1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)        
            print(comb)
    return all_comb


def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And 
    generates a kernel that represents a line intersection, where the center 
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")
    match = [(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0],match[m][1]] = 1
        kernels.append(tmp)
    return kernels

            
def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4,3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list,taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels


# Find the curve intersections
def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve 
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels 
                       are the line intersection.
    """
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        output_image = output_image + out
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
        show_image[:,:,1] = show_image[:,:,1] -  output_image *255
        show_image[:,:,2] = show_image[:,:,2] -  output_image *255
        plt.imshow(show_image)
    return output_image


# Thinning function  
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    """
    the Zhang-Suen Thinning Algorithm
    INPUT:
        image - Bidimesional (m x n) np.array of the image with bool dtype
    OUTPUT:
        Image_Thinned - Bidimesional (m x n) np.array of bool dtype
    
    """
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

# 
def find_endoflines(input_image, show=0):
    """
    """
    kernel_0 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1, 1, -1]), dtype="int")
    
    kernel_1 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [1,-1, -1]), dtype="int")
    
    kernel_2 = np.array((
            [-1, -1, -1],
            [1, 1, -1],
            [-1,-1, -1]), dtype="int")
    
    kernel_3 = np.array((
            [1, -1, -1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    
    kernel_4 = np.array((
            [-1, 1, -1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    
    kernel_5 = np.array((
            [-1, -1, 1],
            [-1, 1, -1],
            [-1,-1, -1]), dtype="int")
    
    kernel_6 = np.array((
            [-1, -1, -1],
            [-1, 1, 1],
            [-1,-1, -1]), dtype="int")
    
    kernel_7 = np.array((
            [-1, -1, -1],
            [-1, 1, -1],
            [-1,-1, 1]), dtype="int")
    
    kernel = np.array((kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i,:,:])
        output_image = output_image + out
        
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
        show_image[:,:,1] = show_image[:,:,1] -  output_image *255
        show_image[:,:,2] = show_image[:,:,2] -  output_image *255
        plt.imshow(show_image)    
        
    return output_image#, np.where(output_image == 1)

    
        
 


###############################################################################
#  PIPELINE
###############################################################################     
# 0 - Load the image in grayscale
img = cv2.imread('sample1.jpg',cv2.IMREAD_GRAYSCALE)
inv_img = (255-img)




# 1 - IMAGE PROCESSING

# APPLY OTSU THRESHOLDING must set object region as 1, background region as 0 !
#otsu_thres = threshold_otsu(img)  
#bw_img = (inv_img) < otsu_thres    # for some reason it doesnt work well. Must double check

# APPLY simple thresholding
ret,thresh1 = cv2.threshold(inv_img,55,255,cv2.THRESH_BINARY)
bw_img = thresh1.astype(bool)
# Do some thining

# Perform the distance transform algorithm
#dist = cv2.distanceTransform(thresh1, cv2.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
#cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

# APPLY ZHANG SUEN THINING 
#BW_Skeleton = zhangSuen(bw_img)
BW_Skeleton = skeletonize(bw_img)
# BW_Skeleton = BW_Original



# 1.1 - DISPLAY THINNING RESULT
fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
ax1.imshow(bw_img, cmap=plt.cm.gray)
ax1.set_title('Original binary image')
ax1.axis('off')
ax2.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax2.set_title('Skeleton of the image')
ax2.axis('off')
plt.show()

# 2 - EXTRACT THE GRAPH FROM IMAGE

# 2.1 - Find end of lines
input_image = BW_Skeleton.astype(np.uint8)
eol_img = find_endoflines(input_image, 0)

# 2.2- Find curve Intersections
lint_img = find_line_intersection(input_image, 0)

# 2.3 - Put together all the nodes
nodes = eol_img + lint_img
#plt.imshow(nodes)

# 2.4- Join nodes that are so close that could be considered just one node
jointed_nodes = cv2.dilate(nodes,np.ones((3,3)))
#plt.imshow(jointed_nodes)
ret, labeled_nodes = cv2.connectedComponents(jointed_nodes.astype(np.uint8))
output = cv2.connectedComponentsWithStats(jointed_nodes.astype(np.uint8), 8, cv2.CV_32S)
labeled_nodes = output[1]
centroid_nodes =output[3]
#visualize_labeled_connected_components(labeled_nodes)

# 4- subtract nodePixels from the skeleton image
segmented_skeleton = input_image-jointed_nodes
segmented_skeleton[segmented_skeleton<=0] = 0
#plt.imshow(segmented_skeleton)

# 5- detect the connected components
ret, edge_labels = cv2.connectedComponents(segmented_skeleton.astype(np.uint8))
#visualize_labeled_connected_components(edge_labels)

# 6- For each connected component. look for 2 nodes that have overlaping pixels
jointed_nodes = cv2.dilate(nodes,np.ones((5,5)))
output = cv2.connectedComponentsWithStats(jointed_nodes.astype(np.uint8), 8, cv2.CV_32S)
labeled_nodes = output[1]

node2node_edge = [] 
edge_length = []
for edge_n in np.arange(1,np.max(edge_labels)+1):
    visited = 0
    edges_node = []
    for node_n in np.arange(1,np.max(labeled_nodes)+1):
        base = np.zeros_like(jointed_nodes)
        base[edge_labels==edge_n] = 1
        base2 = np.zeros_like(jointed_nodes)
        base2[labeled_nodes==node_n] = 1
        intersec = base2 + base
        match_max = np.max(intersec)
        if match_max == 2:
            visited = visited +1
            edges_node.append(node_n)#np.where(intersec == 2))
            if visited >= 2:
                node2node_edge.append(edges_node)
                edge_length.append(np.sum(base > 0))
                break

# Visualize the edges with an overlay
centroids = output[3]
fil = np.copy(edge_labels)
for i in node2node_edge:
    cv2.line(fil,(int(centroids[i[0]][0]),int(centroids[i[0]][1])),(int(centroids[i[1]][0]),int(centroids[i[1]][1])),(10,10,0), 1)
fil = fil + jointed_nodes*25
plt.imshow(fil)

# Create the network graph
G = nx.Graph()
G.add_nodes_from(np.arange(1,np.max(labeled_nodes)+1))
for n, edge in enumerate(node2node_edge):
    G.add_edge(edge[0], edge[1],length=edge_length[n] )
nx.draw(G, with_labels=True, font_weight='bold')
        

    

"""
# Find Edges Algorithm
0- slightly dilate the node pixels
1- Delete the node pixels from the thinned network image
2- Detect connected component.
3- For each connected component. look for 2 nodes that have overlaping pixels
4- Create the graph in NetworkX
"""


# playing with skeletons
'''
img = cv2.imread('man.jpg',cv2.IMREAD_GRAYSCALE)

ret, t = cv2.threshold(img,200,255 ,cv2.THRESH_BINARY_INV)
bool_img = t.astype(bool)
thin = zhangSuen(bool_img)
thin= thin.astype(np.uint8)
plt.imshow(thin)
plt.show()
'''


