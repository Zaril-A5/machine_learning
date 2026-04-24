"""
graph.py — stores graph data and reads the problem file.

the input file should look like this:
    Nodes:
    1: (4,1)
    Edges:
    (2,1): 4
    Origin:
    2
    Destinations:
    5; 4
"""

import math
import re
from typing import Dict, List, Tuple, Set


class Graph:
    """
    represents a weighted directed graph for the route finding problem.

    Attributes:
        nodes              : {node_id -> (x, y)} — stores x,y positions
        edges              : {node_id -> [(neighbor_id, cost), ...]} — connections
        origin             : starting node ID
        destinations       : set of goal node IDs we're trying to reach
    """

    def __init__(self):
        # initialize all the graph data structures
        self.nodes:        Dict[int, Tuple[float, float]]  = {}
        self.edges:        Dict[int, List[Tuple[int, float]]] = {}
        self.origin:       int | None = None
        self.destinations: Set[int]   = set()

    # === building up the graph ===

    def adding_node(self, node_id: int, x_coordinate: float, y_coordinate: float) -> None:
        # store a node with its x,y position on the map
        self.nodes[node_id] = (x_coordinate, y_coordinate)

    def adding_edge(self, from_node: int, to_node: int, edge_cost: float) -> None:
        # add a one-way path from from_node to to_node with a cost value
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, edge_cost))

    def prepare_graph_for_search(self) -> None:
        # call this once after adding all nodes and edges
        # it checks that all edges point to real nodes
        # and sorts edges so smaller node ids are tried first
        for source_node, adjacent_nodes in list(self.edges.items()):
            for target_node, _ in adjacent_nodes:
                if target_node not in self.nodes:
                    raise ValueError(f"Edge ({source_node},{target_node}) references unknown node {target_node}")
            # sort so smaller node ids get expanded first
            self.edges[source_node] = sorted(adjacent_nodes, key=lambda edge_tuple: edge_tuple[0])

    # === methods for looking up info in the graph ===

    def getting_successors(self, current_node: int) -> List[Tuple[int, float]]:
        # get all neighbors of a node and their edge costs, sorted by node id
        return list(self.edges.get(current_node, []))

    def calculating_heuristic(self, current_node: int) -> float:
        # estimate how far we are from the nearest destination using straight-line distance
        # this helps informed search algorithms like A* make better decisions
        if current_node not in self.nodes or not self.destinations:
            return float('inf')
        
        current_x, current_y = self.nodes[current_node]
        closest_distance = float('inf')
        
        # check distance to each destination and find the nearest one
        for destination_node in self.destinations:
            destination_x, destination_y = self.nodes[destination_node]
            straight_line_distance = math.hypot(current_x - destination_x, current_y - destination_y)
            if straight_line_distance < closest_distance:
                closest_distance = straight_line_distance
        
        return closest_distance

    def checking_if_goal(self, current_node: int) -> bool:
        # return true if we reached a destination node
        return current_node in self.destinations


# ======================================================================
# file parser — reads the problem from a text file
# ======================================================================

def parsing_graph_file(filename: str) -> Graph:
    """
    reads a problem file and builds a graph from it.
    the file can have sections in any order: Nodes, Edges, Origin, Destinations
    """
    graph_object = Graph()
    current_section = None

    # regex patterns to match node and edge lines
    number_pattern     = r'-?\d+(?:\.\d+)?'
    node_pattern = re.compile(rf'^\s*(\d+)\s*:\s*\(\s*({number_pattern})\s*,\s*({number_pattern})\s*\)\s*$')
    edge_pattern = re.compile(rf'^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*({number_pattern})\s*$')

    # open and read the file line by line
    with open(filename, 'r') as file_handle:
        for raw_line in file_handle:
            current_line = raw_line.strip()
            if not current_line:
                continue

            lowercase_line = current_line.lower()

            # ---- check for section headers ----
            if lowercase_line.startswith('nodes:'):
                current_section = 'nodes'
                continue

            if lowercase_line.startswith('edges:'):
                current_section = 'edges'
                continue

            if lowercase_line.startswith('origin:'):
                # sometimes origin is on the same line as the header
                remaining_text = current_line[len('Origin:'):].strip()
                if remaining_text:
                    graph_object.origin = int(remaining_text)
                    current_section = None
                else:
                    current_section = 'origin'
                continue

            if lowercase_line.startswith('destinations:'):
                # destinations can be on the same line (separated by semicolons)
                remaining_text = current_line[len('Destinations:'):].strip()
                if remaining_text:
                    for dest_part in remaining_text.split(';'):
                        dest_part = dest_part.strip()
                        if dest_part:
                            graph_object.destinations.add(int(dest_part))
                    current_section = None
                else:
                    current_section = 'destinations'
                continue

            # ---- process data lines based on current section ----
            if current_section == 'nodes':
                regex_match = node_pattern.match(current_line)
                if regex_match:
                    graph_object.adding_node(int(regex_match.group(1)),
                                   float(regex_match.group(2)),
                                   float(regex_match.group(3)))
                else:
                    raise ValueError(f"bad node line: '{current_line}'")

            elif current_section == 'edges':
                regex_match = edge_pattern.match(current_line)
                if regex_match:
                    graph_object.adding_edge(int(regex_match.group(1)),
                                   int(regex_match.group(2)),
                                   float(regex_match.group(3)))
                else:
                    raise ValueError(f"bad edge line: '{current_line}'")

            elif current_section == 'origin':
                graph_object.origin = int(current_line)
                current_section = None

            elif current_section == 'destinations':
                for dest_part in current_line.split(';'):
                    dest_part = dest_part.strip()
                    if dest_part:
                        graph_object.destinations.add(int(dest_part))
                current_section = None

    graph_object.prepare_graph_for_search()
    return graph_object


# ======================================================================
# run this to make sure the file reader works
# ======================================================================

if __name__ == '__main__':
    # test data — the sample graph from the assignment
    sample_graph_text = """Nodes:
1: (4,1)
2: (2,2)
3: (4,4)
4: (6,3)
5: (5,6)
6: (7,5)
Edges:
(2,1): 4
(3,1): 5
(1,3): 5
(2,3): 4
(3,2): 5
(4,1): 6
(1,4): 6
(4,3): 5
(3,5): 6
(5,3): 6
(4,5): 7
(5,4): 8
(6,3): 7
(3,6): 7
Origin: 2
Destinations: 5; 4
"""
    # write test data to a temporary file
    with open('_smoke_test.txt', 'w') as temp_file:
        temp_file.write(sample_graph_text)

    # parse it and print some info to verify it works
    test_graph = parsing_graph_file('_smoke_test.txt')
    print("Nodes      :", test_graph.nodes)
    print("Origin     :", test_graph.origin)
    print("Destinations:", test_graph.destinations)
    print("Successors(2):", test_graph.getting_successors(2))
    print("h(2)       :", round(test_graph.calculating_heuristic(2), 4))

    # clean up the temp file
    import os
    os.remove('_smoke_test.txt')