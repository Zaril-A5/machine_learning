"""
route_finder.py

Modified A* search algorithm that uses dynamic travel time costs
and returns top-k paths (top-5 paths for this assignment).

Author: Member 1
"""

import heapq
from typing import List, Tuple, Optional, Callable
from datetime import datetime


class PathNode:
    """
    Represents a node in the search tree during pathfinding.
    Different from graph nodes - this is for search exploration.
    """
    
    def __init__(self, site_id: int, parent: Optional['PathNode'] = None, 
                 g_cost: float = 0.0, h_cost: float = 0.0):
        """
        Initialize path node.
        
        Parameters:
        - site_id (int): SCATS site at this position
        - parent (PathNode): node we came from
        - g_cost (float): actual cost from start to here
        - h_cost (float): heuristic estimate from here to goal
        """
        self.site_id = site_id
        self.parent = parent
        self.g_cost = g_cost  # actual cost
        self.h_cost = h_cost  # heuristic estimate
        self.f_cost = g_cost + h_cost  # total estimate

    def reconstruct_path(self) -> List[int]:
        """Walk back through parents to reconstruct the path."""
        path = []
        current = self
        while current is not None:
            path.append(current.site_id)
            current = current.parent
        return list(reversed(path))

    def __lt__(self, other):
        """For heap ordering: compare by f_cost, then g_cost."""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return self.g_cost < other.g_cost


class TopKRouteFinder:
    """
    A* pathfinding algorithm modified to find top-k routes.
    Uses dynamic travel time costs instead of static distances.
    """

    def __init__(self, graph, route_finder, timestamp: Optional[str] = None):
        """
        Initialize route finder.
        
        Parameters:
        - graph (GraphBuilder): the graph structure
        - route_finder (TrafficRouteFinder): provides dynamic travel times
        - timestamp (str): departure time (None = current time)
        """
        self.graph = graph
        self.route_finder = route_finder
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")

    def heuristic(self, current_site: int, goal_site: int) -> float:
        """
        Heuristic for A* - straight-line distance to goal.
        
        Parameters:
        - current_site, goal_site (int): SCATS sites
        
        Returns:
        - estimated distance in kilometers
        """
        if current_site not in self.graph.nodes or goal_site not in self.graph.nodes:
            return 0.0
        
        lat1, lon1 = self.graph.nodes[current_site]
        lat2, lon2 = self.graph.nodes[goal_site]
        
        # Straight-line distance
        distance = self.graph.haversine_distance(lat1, lon1, lat2, lon2)
        return distance

    def find_top_k_paths(self, start_site: int, goal_site: int, 
                         k: int = 5, max_iterations: int = 1000) -> List[List[int]]:
        """
        Find top-k shortest paths from start to goal using modified A*.
        
        Parameters:
        - start_site (int): starting SCATS site
        - goal_site (int): destination SCATS site
        - k (int): number of paths to find (default 5)
        - max_iterations (int): max search iterations to prevent infinite loops
        
        Returns:
        - list of k paths (each path is list of SCATS sites)
        """
        if start_site == goal_site:
            return [[start_site]]
        
        paths_found = []
        open_set = []
        visited_paths = set()  # Track which paths we've explored
        iteration = 0
        
        # Initialize with start node
        start_node = PathNode(start_site, parent=None, g_cost=0.0, 
                             h_cost=self.heuristic(start_site, goal_site))
        heapq.heappush(open_set, (start_node.f_cost, iteration, start_node))
        iteration += 1
        
        while open_set and len(paths_found) < k and iteration < max_iterations:
            _, _, current_node = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current_node.site_id == goal_site:
                path = current_node.reconstruct_path()
                path_tuple = tuple(path)
                
                # Avoid duplicate paths
                if path_tuple not in visited_paths:
                    paths_found.append(path)
                    visited_paths.add(path_tuple)
                continue
            
            # Explore neighbors
            neighbors = self.graph.get_successors(current_node.site_id)
            
            for neighbor_site, distance_km in neighbors:
                # Get dynamic travel time instead of using static distance
                try:
                    travel_time_s = self.route_finder.get_edge_travel_time(
                        current_node.site_id, neighbor_site, self.timestamp
                    )
                    # Convert seconds to hours for cost calculation
                    edge_cost = travel_time_s / 3600.0
                except Exception as e:
                    # Fallback: use distance-based cost if travel time unavailable
                    edge_cost = distance_km / 60.0  # Assume 60 km/h base speed
                
                # Calculate costs for this neighbor
                new_g_cost = current_node.g_cost + edge_cost
                h_cost = self.heuristic(neighbor_site, goal_site)
                
                # Create new node
                neighbor_node = PathNode(
                    neighbor_site,
                    parent=current_node,
                    g_cost=new_g_cost,
                    h_cost=h_cost
                )
                
                heapq.heappush(open_set, (neighbor_node.f_cost, iteration, neighbor_node))
                iteration += 1
        
        return paths_found

    def find_top_5_paths(self, start_site: int, goal_site: int) -> List[Tuple[List[int], float]]:
        """
        Find top-5 paths and their travel times.
        Convenience wrapper around find_top_k_paths.
        
        Parameters:
        - start_site (int): starting SCATS site
        - goal_site (int): destination SCATS site
        
        Returns:
        - list of (path, travel_time) tuples sorted by travel time
        """
        paths = self.find_top_k_paths(start_site, goal_site, k=5)
        
        # Calculate travel time for each path
        results = []
        for path in paths:
            try:
                total_time, _ = self.route_finder.get_travel_times_for_route(
                    path, self.timestamp
                )
                results.append((path, total_time))
            except Exception as e:
                print(f"[WARNING] Could not calculate travel time for path {path}: {e}")
        
        # Sort by travel time
        results.sort(key=lambda x: x[1])
        
        return results
