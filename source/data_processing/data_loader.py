"""
data_loader.py

Loads traffic location data from the provided CSV file.
Extracts SCATS site numbers, names, and lat/lon coordinates.

Author: Member 1
"""

import pandas as pd
from typing import Dict, Tuple, List


class TrafficLocationLoader:
    """
    Loads traffic location data from CSV file provided by lecturer.
    CSV columns include: X (lon), Y (lat), SITE_DESC, and other metadata.
    """

    def __init__(self, csv_path: str):
        """
        Initialize loader with path to traffic location CSV.
        
        Parameters:
        - csv_path (str): path to Traffic_Count_Locations_with_LONG_LAT.csv
        """
        self.csv_path = csv_path
        self.locations_df = None
        self.sites = {}  # {site_id -> (lat, lon, description)}

    def load_locations(self) -> Dict[int, Tuple[float, float, str]]:
        """
        Load traffic locations from CSV file.
        
        Returns:
        - dict mapping site_id -> (latitude, longitude, site_description)
        """
        try:
            # Read CSV file
            self.locations_df = pd.read_csv(self.csv_path)
            
            # Extract relevant columns
            # CSV has: X (longitude), Y (latitude), FID (site ID), SITE_DESC
            for _, row in self.locations_df.iterrows():
                site_id = int(row['FID'])
                latitude = float(row['Y'])
                longitude = float(row['X'])
                description = str(row['SITE_DESC'])
                
                self.sites[site_id] = (latitude, longitude, description)
            
            print(f"[OK] Loaded {len(self.sites)} traffic locations from CSV")
            return self.sites
            
        except FileNotFoundError:
            print(f"[ERROR] CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            raise

    def get_site(self, site_id: int) -> Tuple[float, float, str]:
        """
        Get coordinates for a specific site.
        
        Parameters:
        - site_id (int): SCATS site number
        
        Returns:
        - (latitude, longitude, description)
        """
        if not self.sites:
            raise ValueError("Sites not loaded. Call load_locations() first.")
        
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found in data.")
        
        return self.sites[site_id]

    def get_all_sites(self) -> Dict[int, Tuple[float, float, str]]:
        """
        Get all loaded sites.
        
        Returns:
        - dict mapping site_id -> (latitude, longitude, description)
        """
        if not self.sites:
            raise ValueError("Sites not loaded. Call load_locations() first.")
        
        return self.sites.copy()

    def get_site_ids(self) -> List[int]:
        """Get list of all loaded site IDs."""
        return sorted(list(self.sites.keys()))

    def display_sample(self, count: int = 5) -> None:
        """
        Print sample of loaded locations for debugging.
        
        Parameters:
        - count (int): number of samples to display
        """
        site_ids = self.get_site_ids()[:count]
        
        print(f"\n{'='*80}")
        print(f"Sample of {min(count, len(site_ids))} traffic locations:")
        print(f"{'='*80}")
        print(f"{'Site ID':<10} {'Latitude':<15} {'Longitude':<15} {'Description':<40}")
        print(f"{'-'*80}")
        
        for site_id in site_ids:
            lat, lon, desc = self.sites[site_id]
            print(f"{site_id:<10} {lat:<15.6f} {lon:<15.6f} {desc[:39]:<40}")
        
        print(f"{'='*80}\n")
