"""
traffic_flow_loader.py

Loads SCATS traffic flow data from Scats Data October 2006.xls.

Author: Member 2
"""

import pandas as pd
from typing import Dict, List, Tuple


class TrafficFlowLoader:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.flow_df = None
        self.flow_data = {}

#[site_id, date, location], [timeInterval (hour:minute , example 02:15), value (ie the actual amount of cars that passed at that time interval)]
    def load_flow_data(self) -> Dict[Tuple[int, str, str], Dict[str, int]]:
        try:
            self.flow_df = pd.read_excel(self.excel_path, sheet_name="Data", header=1)
            
            # Time intervals for V00-V95 (96 intervals of 15 minutes)
            time_intervals = []
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    time_intervals.append(f"{hour}:{minute:02d}")
            
            # Process each row
            for _, row in self.flow_df.iterrows():
                site_id = int(row['SCATS Number'])
                date_str = str(row['Date'])
                location = str(row.get('Location', ''))
                
                # Extract flow values for each time interval
                flows = {}
                for i, time_interval in enumerate(time_intervals):
                    col_name = f'V{i:02d}'
                    if col_name in row:
                        flow_value = row[col_name]
                        if pd.notna(flow_value):
                            flows[time_interval] = int(flow_value)
                
                self.flow_data[(site_id, date_str, location)] = flows
            
            print(f"[OK] Loaded {len(self.flow_data)} traffic flow records from Excel")
            return self.flow_data
            
        except FileNotFoundError:
            print(f"[ERROR] Excel file not found: {self.excel_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load Excel: {e}")
            raise

    def get_flow(self, site_id: int, date: str, time_interval: str, location: str) -> int:
        if (site_id, date, location) in self.flow_data:
            return self.flow_data[(site_id, date, location)].get(time_interval, 0)
        raise ValueError(f"No data for site {site_id} on {date} at location {location}")

    def get_site_dates(self, site_id: int) -> List[str]:
        dates = []
        for (sid, date, _) in self.flow_data.keys():
            if sid == site_id:
                dates.append(date)
        return sorted(list(set(dates)))

    def get_all_sites(self) -> List[int]:
        sites = set()
        for (site_id, _, _) in self.flow_data.keys():
            sites.add(site_id)
        return sorted(list(sites))

    def display_sample(self, count: int = 3) -> None:
        samples = list(self.flow_data.items())[:count]
        
        print(f"\n{'='*100}")
        print(f"Sample of {min(count, len(samples))} traffic flow records:")
        print(f"{'='*100}")
        
        for (site_id, date, location), flows in samples:
            print(f"\nSite ID: {site_id}, Date: {date}, Location: {location}")
            print(f"Time intervals with flow data: {len(flows)}")

            sorted_times = sorted(flows.keys(), key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))
            if len(sorted_times) > 10:
                print(f"First 5: {[(t, flows[t]) for t in sorted_times[:5]]}")
                print(f"Last 5:  {[(t, flows[t]) for t in sorted_times[-5:]]}")
            else:
                print(f"All intervals: {[(t, flows[t]) for t in sorted_times]}")
        
        print(f"{'='*100}\n")
