"""
search.py — different search algorithms to find routes in a graph.

you can run this from the command line:
    python search.py <filename> <method>

method options (type any of these):
    DFS   – depth first search          (no heuristic)
    BFS   – breadth first search        (no heuristic)
    GBFS  – greedy best first search    (uses heuristic)
    AS    – a-star search               (uses heuristic)
    CUS1  – iterative deepening dfs     (custom, no heuristic)
    CUS2  – iterative deepening a-star  (custom, uses heuristic)

when a goal is found, it prints:
    <filename> <method>
    <goal_node_id> <number_of_nodes_created>
    <path>          ← the route from start to goal

if no path exists, it prints:
    <filename> <method>
    No solution found.

note about tie-breaking (when nodes have same priority):
    we always expand the one with the smaller id first.
    if ids are tied too, we expand in the order they were added (chronological).
    the code uses heap tuples to enforce this: (priority, state, counter, node)
"""

import heapq
from collections import deque
from graph import parsing_graph_file


# ======================================================================
# SearchTreeNode — represents one position in the search tree
# ======================================================================

class SearchTreeNode:
    """
    represents a node in our search tree (different from nodes in the graph).

    Attributes:
        state              : the graph node id at this position in the tree
        parent             : the SearchTreeNode we came from (None at the start)
        cost               : g(n) — total cost accumulated from start to here
        depth              : how many edges we've traversed from the start
    """
    __slots__ = ('state', 'parent', 'cost', 'depth')

    def __init__(self, state, parent=None, cost=0.0, depth=0):
        self.state  = state
        self.parent = parent
        self.cost   = cost
        self.depth  = depth

    def reconstructing_path(self):
        # walk backwards through parent pointers to the start
        # build a list of all node ids from start to here
        current_tree_node, path_steps = self, []
        while current_tree_node:
            path_steps.append(current_tree_node.state)
            current_tree_node = current_tree_node.parent
        # reverse it so it goes from start to goal
        return list(reversed(path_steps))


# ======================================================================
# 1. DFS — depth first search (no heuristic)
# ======================================================================

def depth_first_searching(search_graph):
    """
    explores by going as deep as possible, then backtracking when stuck (lifo stack).

    how tie-breaking works:
        the graph already sorts neighbors by node id.
        we push them in reverse order onto the stack so the smallest id
        ends up on top and gets explored first.

    how we avoid infinite loops:
        keep a 'visited' set of expanded nodes. if we pop a node that's
        already been visited, we skip it.

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    # start from the origin node
    starting_node = search_graph.origin
    total_nodes_created = 1
    visited_set = set()
    nodes_stack = [SearchTreeNode(starting_node)]

    # keep searching until the stack is empty
    while nodes_stack:
        current_tree_node = nodes_stack.pop()

        # if we've already visited this node, skip it
        if current_tree_node.state in visited_set:
            continue
        visited_set.add(current_tree_node.state)

        # did we reach a destination?
        if search_graph.checking_if_goal(current_tree_node.state):
            return current_tree_node, total_nodes_created, current_tree_node.reconstructing_path()

        # add neighbors to the stack in reverse order so smallest id is on top
        for neighbor_node, edge_cost in reversed(search_graph.getting_successors(current_tree_node.state)):
            if neighbor_node not in visited_set:
                child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                                        current_tree_node.cost + edge_cost,
                                        current_tree_node.depth + 1)
                nodes_stack.append(child_tree_node)
                total_nodes_created += 1

    return None, total_nodes_created, []


# ======================================================================
# 2. BFS — breadth first search (no heuristic)
# ======================================================================

def breadth_first_searching(search_graph):
    """
    explores all nodes at the current depth before going deeper (fifo queue).

    how tie-breaking works:
        children are added to the queue in order by node id (neighbors come sorted).
        so at the same depth, smaller ids get explored first.

    important detail about visited marking:
        we mark nodes as visited when we ADD them to the queue (not when we explore).
        this prevents adding the same node multiple times.

    why use bfs?
        it always finds the goal with the fewest edges (shallowest path).

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    starting_node = search_graph.origin
    total_nodes_created = 1
    visited_set = {starting_node}
    nodes_queue = deque([SearchTreeNode(starting_node)])

    # keep exploring while there are nodes in the queue
    while nodes_queue:
        current_tree_node = nodes_queue.popleft()

        # did we reach a destination?
        if search_graph.checking_if_goal(current_tree_node.state):
            return current_tree_node, total_nodes_created, current_tree_node.reconstructing_path()

        # add neighbors to the queue (they're already sorted by node id)
        for neighbor_node, edge_cost in search_graph.getting_successors(current_tree_node.state):
            if neighbor_node not in visited_set:
                visited_set.add(neighbor_node)
                child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                                        current_tree_node.cost + edge_cost,
                                        current_tree_node.depth + 1)
                nodes_queue.append(child_tree_node)
                total_nodes_created += 1

    return None, total_nodes_created, []


# ======================================================================
# 3. GBFS — greedy best first search (uses heuristic)
# ======================================================================

def greedy_best_first_searching(search_graph):
    """
    always explores the node that looks closest to the goal (greedy!).
    uses only the heuristic h(n) = straight-line distance to nearest destination.

    how the priority queue works:
        heap entry is: (h_value, state, counter, tree_node)
        h_value      : main priority (smaller = higher priority)
        state        : tie-break — expand smaller node id first
        counter      : tie-break — chronological fifo among equal h and state
        tree_node    : the actual SearchTreeNode object

    visited marking:
        we mark nodes as visited when we POP them from the heap.
        this is called "lazy deletion" and is standard for heap-based search.

    important limitation:
        greedy is fast but not guaranteed to find the cheapest path.

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    starting_node = search_graph.origin
    total_nodes_created = 1
    expansion_counter = 0
    visited_set = set()

    start_tree_node = SearchTreeNode(starting_node)
    # priority queue: (heuristic_value, node_state, counter, tree_node)
    open_list = [(search_graph.calculating_heuristic(starting_node), starting_node, expansion_counter, start_tree_node)]

    # keep exploring while there are nodes to check
    while open_list:
        _, _, _, current_tree_node = heapq.heappop(open_list)

        # skip if we've already explored this node
        if current_tree_node.state in visited_set:
            continue
        visited_set.add(current_tree_node.state)

        # did we reach a destination?
        if search_graph.checking_if_goal(current_tree_node.state):
            return current_tree_node, total_nodes_created, current_tree_node.reconstructing_path()

        # add neighbors to the priority queue
        for neighbor_node, edge_cost in search_graph.getting_successors(current_tree_node.state):
            if neighbor_node not in visited_set:
                expansion_counter += 1
                child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                                       current_tree_node.cost + edge_cost,
                                       current_tree_node.depth + 1)
                total_nodes_created += 1
                # push based on heuristic distance only
                heapq.heappush(open_list,
                               (search_graph.calculating_heuristic(neighbor_node),
                                neighbor_node, expansion_counter, child_tree_node))

    return None, total_nodes_created, []


# ======================================================================
# 4. AS — a-star search (uses heuristic, optimal)
# ======================================================================

def a_star_searching(search_graph):
    """
    expands nodes in order of f(n) = g(n) + h(n), where:
        g(n) = actual cost from start to this node
        h(n) = euclidean distance from this node to the nearest destination

    why a-star?
        because h is admissible (never overestimates), a-star is guaranteed
        to find the optimal (cheapest) path.

    priority queue entries are: (f_value, state, counter, tree_node)
        same tie-breaking rules as greedy: smaller state first, then chronological.

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    starting_node = search_graph.origin
    total_nodes_created = 1
    expansion_counter = 0
    visited_set = set()

    start_tree_node = SearchTreeNode(starting_node)
    # priority queue: (f_value, node_state, counter, tree_node)
    open_list = [(search_graph.calculating_heuristic(starting_node), starting_node, expansion_counter, start_tree_node)]

    # keep exploring while there are nodes to check
    while open_list:
        _, _, _, current_tree_node = heapq.heappop(open_list)

        # skip if already explored
        if current_tree_node.state in visited_set:
            continue
        visited_set.add(current_tree_node.state)

        # did we reach a destination?
        if search_graph.checking_if_goal(current_tree_node.state):
            return current_tree_node, total_nodes_created, current_tree_node.reconstructing_path()

        # add neighbors to the priority queue
        for neighbor_node, edge_cost in search_graph.getting_successors(current_tree_node.state):
            if neighbor_node not in visited_set:
                expansion_counter += 1
                child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                                       current_tree_node.cost + edge_cost,
                                       current_tree_node.depth + 1)
                total_nodes_created += 1
                # f(n) = g(n) + h(n)
                f_value = child_tree_node.cost + search_graph.calculating_heuristic(neighbor_node)
                heapq.heappush(open_list, (f_value, neighbor_node, expansion_counter, child_tree_node))

    return None, total_nodes_created, []


# ======================================================================
# 5. CUS1 — iterative deepening depth-first search (custom, no heuristic)
# ======================================================================

def iterative_deepening_dfs(search_graph):
    """
    custom search: iterative deepening dfs (iddfs).

    why use iddfs?
        plain dfs can get stuck in infinite loops in graphs with cycles.
        bfs finds the shallowest goal but uses lots of memory.
        iddfs gives you the best of both:
          • memory usage like dfs (only the current path)
          • guarantee of finding shallowest goal like bfs

    how it works:
        we run a depth-limited dfs with limit = 0, 1, 2, ... until:
          • we find a goal — return it!
          • we can't go deeper anywhere — no solution exists
        a 'path_set' tracks the current path to prevent cycles
        without needing a global visited set.

    we count total nodes created across all depth iterations.

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    starting_node = search_graph.origin
    total_nodes_created = [0]   # use a list so the nested function can update it

    def depth_limited_search(current_tree_node, depth_limit, current_path_set):
        """
        do a depth-limited dfs from current_tree_node.

        returns:
            tree_node object  — goal was found
            'cutoff'          — limit was hit before finding goal
            None              — dead end, no solution here
        """
        total_nodes_created[0] += 1

        # did we reach a destination?
        if search_graph.checking_if_goal(current_tree_node.state):
            return current_tree_node   # found it!

        # are we too deep?
        if current_tree_node.depth >= depth_limit:
            return 'cutoff'

        did_hit_cutoff = False

        # try all neighbors (they're sorted by node id)
        for neighbor_node, edge_cost in search_graph.getting_successors(current_tree_node.state):
            # avoid revisiting nodes on this path
            if neighbor_node in current_path_set:
                continue

            child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                               current_tree_node.cost + edge_cost,
                               current_tree_node.depth + 1)
            current_path_set.add(neighbor_node)
            search_result = depth_limited_search(child_tree_node, depth_limit, current_path_set)
            current_path_set.discard(neighbor_node)

            # check what happened
            if search_result == 'cutoff':
                did_hit_cutoff = True
            elif search_result is not None:
                return search_result   # found — pass it up!

        # return based on whether we hit the limit
        return 'cutoff' if did_hit_cutoff else None

    # we can't need a depth larger than the number of nodes
    maximum_possible_depth = len(search_graph.nodes)

    # try increasing depth limits
    for current_depth_limit in range(maximum_possible_depth + 1):
        start_tree_node = SearchTreeNode(starting_node)
        path_set = {starting_node}
        search_result = depth_limited_search(start_tree_node, current_depth_limit, path_set)

        # did we find the goal?
        if search_result not in ('cutoff', None):
            return search_result, total_nodes_created[0], search_result.reconstructing_path()

    # exhausted all depths — no solution
    return None, total_nodes_created[0], []


# ======================================================================
# 6. CUS2 — iterative deepening a-star (custom, uses heuristic)
# ======================================================================

def iterative_deepening_astar(search_graph):
    """
    custom search: ida* (iterative deepening a-star).

    why use ida*?
        a-star stores the entire open list in memory, which can be a problem
        on large graphs. ida* only keeps the current dfs path in memory
        (just o(depth) space) while still finding the optimal solution.
        this works because euclidean distance never overestimates.

    how it works:
        do a dfs but skip any branches where f(n) = g(n) + h(n) exceeds
        the current threshold.
        
        after each full iteration:
          • if we found the goal — done!
          • otherwise, raise the threshold to the smallest f-value we had to skip
          • repeat with the new threshold

        this guarantees we find the cheapest path with minimal memory.

    we count total nodes created across all iterations.

    returns:
        (goal_tree_node, total_nodes_created, path_list) or (None, count, [])
    """
    starting_node = search_graph.origin
    total_nodes_created = [0]

    def explore_with_threshold(current_tree_node, f_threshold, current_path_set):
        """
        recursive ida* helper. dfs with a budget on f-values.

        returns:
            (-1, found_node)        — goal reached (success signal!)
            (next_f_value, None)    — not found; tells caller new threshold
        """
        total_nodes_created[0] += 1
        f_value = current_tree_node.cost + search_graph.calculating_heuristic(current_tree_node.state)

        # are we over budget?
        if f_value > f_threshold:
            return f_value, None   # report this f-value as a possible new threshold

        # did we reach the goal?
        if search_graph.checking_if_goal(current_tree_node.state):
            return -1, current_tree_node   # -1 means "success!"

        smallest_f_over_threshold = float('inf')

        # try all neighbors (sorted by node id)
        for neighbor_node, edge_cost in search_graph.getting_successors(current_tree_node.state):
            # don't revisit nodes on the current path
            if neighbor_node in current_path_set:
                continue

            child_tree_node = SearchTreeNode(neighbor_node, current_tree_node,
                               current_tree_node.cost + edge_cost,
                               current_tree_node.depth + 1)
            current_path_set.add(neighbor_node)
            f_status, search_result = explore_with_threshold(child_tree_node, f_threshold, current_path_set)
            current_path_set.discard(neighbor_node)

            # check what happened
            if f_status == -1:
                return -1, search_result   # found! pass it up!
            if f_status < smallest_f_over_threshold:
                smallest_f_over_threshold = f_status   # remember this for next iteration

        return smallest_f_over_threshold, None

    # start with threshold = heuristic at the origin
    current_threshold = search_graph.calculating_heuristic(starting_node)

    # keep raising the threshold until we find the goal
    while True:
        start_tree_node = SearchTreeNode(starting_node)
        path_set = {starting_node}
        f_status, search_result = explore_with_threshold(start_tree_node, current_threshold, path_set)

        # did we find the goal?
        if f_status == -1:
            return search_result, total_nodes_created[0], search_result.reconstructing_path()
        # did we exhaust all possibilities?
        if f_status == float('inf'):
            return None, total_nodes_created[0], []
        # raise threshold and try again
        current_threshold = f_status


# ======================================================================
# command line interface — run this to test a search method on a file
# ======================================================================

if __name__ == '__main__':
    import sys

    # check that we have the right number of arguments
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    input_filename = sys.argv[1]
    search_method = sys.argv[2].upper()

    # map method names to functions
    method_map = {
        'DFS':  depth_first_searching,
        'BFS':  breadth_first_searching,
        'GBFS': greedy_best_first_searching,
        'AS':   a_star_searching,
        'CUS1': iterative_deepening_dfs,
        'CUS2': iterative_deepening_astar,
    }

    # is this a valid method?
    if search_method not in method_map:
        print(f"Unknown method: {search_method}")
        print("Valid methods: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)

    # load the graph from the file
    loaded_graph = parsing_graph_file(input_filename)
    goal_tree_node, total_nodes_made, final_path = method_map[search_method](loaded_graph)

    # output the results
    if goal_tree_node is None:
        # no solution found
        print(f"{input_filename} {search_method}")
        print("No solution found.")
    else:
        # found a path!
        path_string = ' '.join(str(node_id) for node_id in final_path)
        print(f"{input_filename} {search_method}")
        print(f"{goal_tree_node.state} {total_nodes_made}")
        print(path_string)
