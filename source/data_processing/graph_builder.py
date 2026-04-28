"""
graph_builder.py

Builds the Boroondara graph structure using SCATS site numbers as nodes.
Calculates distances between connected intersections using Haversine formula.

Author: Member 1
"""

import math
from typing import Dict, List, Tuple, Set
from data_loader import TrafficLocationLoader


class GraphBuilder:
    """
    Builds a graph for the Boroondara area where:
    - Nodes are SCATS site numbers
    - Edges represent connections between intersections
    - Edge weights are distances calculated using Haversine formula
    """

    def __init__(self, locations_loader: TrafficLocationLoader):
        """
        Initialize graph builder.
        
        Parameters:
        - locations_loader (TrafficLocationLoader): loader with site coordinates
        """
        self.loader = locations_loader
        self.nodes: Dict[int, Tuple[float, float]] = {}  # {site_id -> (lat, lon)}
        self.edges: Dict[int, List[Tuple[int, float]]] = {}  # {from_site -> [(to_site, distance_km), ...]}
        self.site_descriptions: Dict[int, str] = {}

    def add_nodes_from_locations(self) -> None:
        """
        Add all SCATS sites as graph nodes.
        Uses coordinates from loaded traffic location data.
        """
        sites = self.loader.get_all_sites()
        
        for site_id, (latitude, longitude, description) in sites.items():
            self.nodes[site_id] = (latitude, longitude)
            self.site_descriptions[site_id] = description
            # Initialize empty edge list for each node
            if site_id not in self.edges:
                self.edges[site_id] = []
        
        print(f"[OK] Added {len(self.nodes)} nodes to graph")

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Parameters:
        - lat1, lon1: first point (latitude, longitude in degrees)
        - lat2, lon2: second point (latitude, longitude in degrees)
        
        Returns:
        - distance in kilometers
        """
        # Earth's radius in kilometers
        EARTH_RADIUS_KM = 6371.0
        
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance_km = EARTH_RADIUS_KM * c
        return distance_km

    def add_edge(self, from_site: int, to_site: int) -> None:
        """
        Add a directed edge between two sites.
        Calculates distance using Haversine formula.
        
        Parameters:
        - from_site (int): starting SCATS site
        - to_site (int): destination SCATS site
        """
        if from_site not in self.nodes:
            raise ValueError(f"Source site {from_site} not in graph")
        if to_site not in self.nodes:
            raise ValueError(f"Destination site {to_site} not in graph")
        
        lat1, lon1 = self.nodes[from_site]
        lat2, lon2 = self.nodes[to_site]
        
        distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Check if edge already exists (avoid duplicates)
        for i, (existing_to, _) in enumerate(self.edges[from_site]):
            if existing_to == to_site:
                self.edges[from_site][i] = (to_site, distance_km)
                return
        
        self.edges[from_site].append((to_site, distance_km))

    def add_bidirectional_edge(self, site_a: int, site_b: int) -> None:
        """
        Add edges in both directions between two sites (road can be traveled both ways).
        
        Parameters:
        - site_a, site_b: SCATS sites to connect
        """
        self.add_edge(site_a, site_b)
        self.add_edge(site_b, site_a)

    def add_edges_between_nearby_sites(self, distance_threshold_km: float = 3.0) -> None:
        """
        Connect nearby sites with edges (automatic connectivity).
        This simplification assumes adjacent intersections within threshold are connected.
        
        IMPORTANT: In production, you would use an actual street network dataset.
        This is a simplified approach for the Boroondara area.
        
        Parameters:
        - distance_threshold_km (float): connect sites within this distance
        """
        site_ids = sorted(self.nodes.keys())
        edge_count = 0
        
        print(f"\n[BUILDING] Connecting nearby sites (threshold: {distance_threshold_km} km)...")
        
        for i, site_a in enumerate(site_ids):
            for site_b in site_ids[i+1:]:
                lat1, lon1 = self.nodes[site_a]
                lat2, lon2 = self.nodes[site_b]
                
                distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)
                
                if 0 < distance_km <= distance_threshold_km:
                    self.add_bidirectional_edge(site_a, site_b)
                    edge_count += 1
        
        print(f"[OK] Added {edge_count} edges connecting nearby sites")

    def get_successors(self, site_id: int) -> List[Tuple[int, float]]:
        """
        Get all neighbors and their distances from a site.
        
        Parameters:
        - site_id (int): SCATS site
        
        Returns:
        - list of (neighbor_id, distance_km)
        """
        if site_id not in self.edges:
            return []
        return self.edges[site_id]

    def get_edge_distance(self, from_site: int, to_site: int) -> float:
        """
        Get distance between two sites.
        
        Parameters:
        - from_site, to_site (int): SCATS sites
        
        Returns:
        - distance in kilometers, or None if no edge
        """
        for neighbor, distance in self.edges.get(from_site, []):
            if neighbor == to_site:
                return distance
        return None

    def get_graph_info(self) -> Dict:
        """
        Get basic information about the graph.
        
        Returns:
        - dict with node count, edge count, etc.
        """
        total_edges = sum(len(neighbors) for neighbors in self.edges.values())
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': total_edges,
            'avg_edges_per_node': total_edges / len(self.nodes) if self.nodes else 0
        }

    def display_graph_info(self) -> None:
        """Print graph statistics."""
        info = self.get_graph_info()
        
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Number of nodes (SCATS sites): {info['num_nodes']}")
        print(f"Number of edges (connections):  {info['num_edges']}")
        print(f"Avg edges per node:             {info['avg_edges_per_node']:.2f}")
        print(f"{'='*60}\n")
