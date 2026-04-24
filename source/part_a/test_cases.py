"""
test_cases.py — 15 tests for all the route finding search algorithms.
how to run:
    python test_cases.py
what it does:
    each test creates a temporary graph file, runs one or more algorithms,
    and checks if the result is correct. prints a pass/fail report at the end.
what it checking for ??:
  correct goal node or none if no there are no solution like disconnected graph
  path starts at origin and ends at a destination
  every step in the path uses a real edge in the graph
  for deterministic algorithms, it checks the exact path match
"""

import os
import tempfile

from graph import parsing_graph_file
from search import (depth_first_searching, breadth_first_searching,
                    greedy_best_first_searching, a_star_searching,
                    iterative_deepening_dfs, iterative_deepening_astar)

# ======================================================================
# helper functions for testing purposes
# ======================================================================

def creating_temp_graph_file(content: str):
    # write graph data to a temporary file and parse it
    # returns the filename and the parsed Graph object
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                   delete=False, encoding='utf-8')
    temp_file.write(content)
    temp_file.close()
    return temp_file.name, parsing_graph_file(temp_file.name)


def checking_if_path_valid(test_graph, route_path: list) -> bool:
    # check that the path uses only real edges in the graph
    # every two consecutive nodes must be connected
    if not route_path:
        return False
    
    for i in range(len(route_path) - 1):
        current_node = route_path[i]
        next_node = route_path[i + 1]
        
        # get all neighbors of current_node
        neighbor_list = [neighbor_id for neighbor_id, _ in test_graph.getting_successors(current_node)]
        
        # is next_node a real neighbor?
        if next_node not in neighbor_list:
            return False
    
    return True


def running_single_test(test_number: int,
             test_description: str,
             test_graph,
             search_function,
             expected_goal_node=None,
             expected_path_list=None,
             expect_no_solution: bool = False) -> bool:
    """
    run one test and report whether it pass or fail.
    parameters:
    - expected_goal_node : if given, check that we reached this node
    - expected_path_list : if given, check that the path exactly matches this
    - expect_no_solution : if true, the test passes only when no path exists
    """
    found_goal_node, total_nodes_created, returned_path = search_function(test_graph)

    # check if the test passed
    if expect_no_solution:
        test_passed = (found_goal_node is None)
    else:
        if found_goal_node is None:
            test_passed = False
        else:
            # check: is the goal node correct?
            goal_node_ok = (expected_goal_node is None) or (found_goal_node.state == expected_goal_node)
            # check: is the path correct?
            path_ok = (expected_path_list is None) or (returned_path == expected_path_list)
            # check: does the path end at a goal?
            ends_at_goal = test_graph.checking_if_goal(returned_path[-1]) if returned_path else False
            # check: does the path use real edges?
            path_uses_real_edges = checking_if_path_valid(test_graph, returned_path)
            # all checks must pass
            test_passed = goal_node_ok and path_ok and ends_at_goal and path_uses_real_edges

    # Prepare result output
    result_status = "PASS" if test_passed else "FAIL"
    goal_state = found_goal_node.state if found_goal_node else "No Solution"
    nodes_count = total_nodes_created
    path_str = str(returned_path) if returned_path else "[]"
    
    # Print result with format: <goal> <nodes> <path>
    print(f"  Test {test_number:02d}: [{result_status}]  {test_description}")
    print(f"           → {goal_state} {nodes_count} {path_str}")

    # if failed, show details
    if not test_passed:
        print(f"           Expected: goal={expected_goal_node}, path={expected_path_list},"
              f" no_solution={expect_no_solution}")
        actual_goal_state = found_goal_node.state if found_goal_node else None
        print(f"           Got     : goal={actual_goal_state}, path={returned_path},"
              f" nodes_created={total_nodes_created}")

    return test_passed


# ======================================================================
# Graph definitions
# ======================================================================

# --- sample graph from the assignment brief ---
SAMPLE = """Nodes:
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

# --- Linear chain: 2→3→4→5→6, dest={4;5} ---
LINEAR = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (3,0)
5: (4,0)
6: (5,0)
Edges:
(2,3): 1
(3,4): 1
(4,5): 1
(5,6): 1
Origin: 2
Destinations: 4; 5
"""

# --- Disconnected: origin=2 cannot reach destinations 4,5 ---
DISCONNECTED = """Nodes:
1: (0,0)
2: (1,0)
3: (5,0)
4: (6,0)
5: (7,0)
Edges:
(2,3): 1
(3,1): 1
(4,5): 1
Origin: 2
Destinations: 4; 5
"""

# --- Immediate goal: destination 4 is adjacent to origin 2 (1 step) ---
ORIGIN_IS_GOAL = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (2,1)
5: (3,1)
Edges:
(2,4): 1
(2,3): 2
(3,5): 1
Origin: 2
Destinations: 4; 5
"""

# --- Two destinations: 4 (closer) and 5 (farther) ---
MULTI_DEST = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (1,1)
5: (2,1)
Edges:
(2,3): 1
(3,5): 1
(2,4): 1
Origin: 2
Destinations: 4; 5
"""

# --- Single edge: minimal graph with just 2→4 ---
SINGLE_EDGE = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (1,1)
5: (2,1)
Edges:
(2,4): 5
(2,3): 3
(3,5): 2
Origin: 2
Destinations: 4; 5
"""

# --- Cyclic graph with multiple paths: 2→3→4 and 2→1→4, plus 5 ---
CYCLIC = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (2,1)
5: (1,1)
Edges:
(2,3): 1
(3,4): 1
(2,1): 1
(1,4): 1
(2,5): 1
Origin: 2
Destinations: 4; 5
"""

# --- Larger graph: two routes to node 4, cheaper via 6 steps vs 3 steps ---
LARGER = """Nodes:
1: (0,0)
2: (2,0)
3: (4,0)
4: (4,2)
5: (2,2)
6: (0,2)
Edges:
(2,3): 3
(3,4): 3
(2,5): 1
(5,6): 1
(6,1): 1
(1,4): 1
Origin: 2
Destinations: 4; 5
"""

# --- Tie-breaking: equal heuristic distances for nodes 3,5 from goal 4 ---
TIE = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (2,1)
5: (1,1)
Edges:
(2,3): 1
(3,4): 1
(2,5): 1
(5,4): 1
Origin: 2
Destinations: 4; 5
"""

# --- Only one path: 2→3→4 and 2→1→5 (no branches, deterministic) ---
ONLY_ONE_PATH = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (3,0)
5: (0,1)
Edges:
(2,3): 2
(3,4): 3
(2,1): 1
(1,5): 4
Origin: 2
Destinations: 4; 5
"""

# --- Dead-end branch: node 3 is a dead end, correct path via 1 ---
DEAD_END = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (2,1)
5: (0,1)
Edges:
(2,3): 1
(2,1): 1
(1,4): 1
(2,5): 1
Origin: 2
Destinations: 4; 5
"""

# --- Star graph: hub (node 2) with multiple branches ---
STAR_GRAPH = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (1,1)
5: (1,-1)
Edges:
(2,3): 1
(2,4): 1
(2,5): 1
(3,4): 2
Origin: 2
Destinations: 4; 5
"""

# --- Diamond: two paths converge at node 3, then to goals 4,5 ---
DIAMOND = """Nodes:
1: (0,0)
2: (1,0)
3: (2,1)
4: (3,1)
5: (2,2)
Edges:
(2,1): 1
(2,3): 2
(1,3): 1
(3,4): 1
(3,5): 1
Origin: 2
Destinations: 4; 5
"""

# --- Misleading heuristic: straight-line distance misleads the search ---
MISLEADING_HEURISTIC = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (3,0)
5: (1,1)
Edges:
(2,5): 1
(5,4): 2
(2,3): 1
(3,4): 3
Origin: 2
Destinations: 4; 5
"""

# --- Spiral topology: cyclic path with shortcuts ---
SPIRAL = """Nodes:
1: (0,0)
2: (1,0)
3: (2,0)
4: (2,1)
5: (1,1)
6: (0,1)
Edges:
(2,3): 1
(3,4): 1
(4,5): 1
(5,6): 1
(6,1): 1
(2,5): 2
(5,4): 1
Origin: 2
Destinations: 4; 5
"""


# ======================================================================
# Optimized test runner — 15 graphs × 6 algorithms = 90 tests
# ======================================================================

# Define all 15 graph types and their descriptions
# Format: (name, content, description, expect_no_solution)
GRAPH_DEFINITIONS = [
    ("SAMPLE", SAMPLE, "Basic graph from assignment", False),
    ("LINEAR", LINEAR, "Simple chain 2→3→4→5→6", False),
    ("DISCONNECTED", DISCONNECTED, "Two separate components (unreachable goals)", True),
    ("ORIGIN_IS_GOAL", ORIGIN_IS_GOAL, "Immediate goal adjacent to origin", False),
    ("MULTI_DEST", MULTI_DEST, "Two destinations at different depths", False),
    ("SINGLE_EDGE", SINGLE_EDGE, "Minimal graph with few edges", False),
    ("CYCLIC", CYCLIC, "Multiple paths with cycles", False),
    ("LARGER", LARGER, "Larger graph with alternate routes", False),
    ("TIE", TIE, "Equal heuristic distances (tie-breaking)", False),
    ("ONLY_ONE_PATH", ONLY_ONE_PATH, "Deterministic single paths", False),
    ("DEAD_END", DEAD_END, "Branch with dead end", False),
    ("STAR_GRAPH", STAR_GRAPH, "Hub topology with radiating branches", False),
    ("DIAMOND", DIAMOND, "Two paths converging at node", False),
    ("MISLEADING_HEURISTIC", MISLEADING_HEURISTIC, "Heuristic conflicts with optimal path", False),
    ("SPIRAL", SPIRAL, "Complex cyclic topology with shortcuts", False),
]

# Define all 6 algorithms and their names
ALGORITHM_DEFINITIONS = [
    ("DFS", depth_first_searching, "Depth-First Search"),
    ("BFS", breadth_first_searching, "Breadth-First Search"),
    ("GBFS", greedy_best_first_searching, "Greedy Best-First Search"),
    ("A*", a_star_searching, "A* Search"),
    ("IDDFS", iterative_deepening_dfs, "Iterative Deepening DFS"),
    ("IDA*", iterative_deepening_astar, "Iterative Deepening A*"),
]


def running_all_tests():
    test_results = []
    test_counter = 1
    
    # Main loop: for each graph, test all 6 algorithms
    for graph_idx, (graph_name, graph_content, graph_desc, expect_no_solution) in enumerate(GRAPH_DEFINITIONS, 1):
        _, test_graph = creating_temp_graph_file(graph_content)
        
        print(f"\n{'='*70}")
        print(f"Graph {graph_idx}: {graph_name:20} — {graph_desc}")
        print(f"{'='*70}")
        
        # Test this graph with all 6 algorithms
        for algo_idx, (algo_short, algo_func, algo_full) in enumerate(ALGORITHM_DEFINITIONS, 1):
            test_desc = f"{algo_full} on {graph_name}"
            
            result = running_single_test(
                test_counter,
                test_desc,
                test_graph,
                algo_func,
                expect_no_solution=expect_no_solution
            )
            
            test_results.append(result)
            test_counter += 1

    # === Print Final Summary ===
    total_tests = len(test_results)
    tests_passed = sum(test_results)
    tests_failed = total_tests - tests_passed
    
    print("\n")
    print("=" * 70)
    print(f"FINAL RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)
    print(f"  Total Tests Run     : {total_tests}")
    print(f"  Graphs Tested       : {len(GRAPH_DEFINITIONS)}")
    print(f"  Algorithms Used     : {len(ALGORITHM_DEFINITIONS)}")
    print(f"  Tests Passed        : {tests_passed}")
    print(f"  Tests Failed        : {tests_failed}")
    
    if tests_passed == total_tests:
        print("\n  ✓ ALL TESTS PASSED!")
    else:
        print(f"\n  ✗ {tests_failed} test(s) FAILED")
    print("=" * 70)


# ======================================================================
# here is entry point — run this to test everything
# ======================================================================

if __name__ == '__main__':
    print("Running 15 test cases...")
    print("-" * 55)
    running_all_tests()