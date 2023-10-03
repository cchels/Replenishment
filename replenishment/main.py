import random
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

class DataGenerator:
    def __init__(self, num_items, num_stores, num_warehouses):
        self.num_items = num_items
        self.num_stores = num_stores
        self.num_warehouses = num_warehouses

    @staticmethod
    def _generate_coordinates(num_points):
        return [random.uniform(0, 90) for _ in range(num_points)], [random.uniform(0, 180) for _ in range(num_points)]

    def generate_random_data(self, mean_demand, demand_std_dev, max_capacity, mean_supply, supply_std_dev,
                             sales_mean, sales_std_dev, offer_mean, offer_std_dev):
        stores_data = self._generate_random_stores_data(mean_demand, demand_std_dev)
        warehouses_data = self._generate_random_warehouses_data(mean_supply, supply_std_dev, max_capacity)
        past_store_records = self._generate_random_past_store_records(sales_mean, sales_std_dev)
        past_warehouse_records = self._generate_random_past_warehouse_records(offer_mean, offer_std_dev)

        return stores_data, warehouses_data, past_store_records, past_warehouse_records

    def _generate_random_stores_data(self, mean_demand, demand_std_dev):
        items = [f'Item{i}' for i in range(1, self.num_items + 1)]
        demands = {
            item: np.clip(np.random.normal(mean_demand, demand_std_dev, self.num_stores).astype(int), 0, None)
            for item in items
        }
        latitudes, longitudes = self._generate_coordinates(self.num_stores)
        stores_data = {
            'StoreID': [f'Store{i}' for i in range(1, self.num_stores + 1)],
            'Latitude': latitudes,
            'Longitude': longitudes,
            **demands
        }
        return pd.DataFrame(stores_data)

    def _generate_random_warehouses_data(self, mean_supply, supply_std_dev, max_capacity):
        items = [f'Item{i}' for i in range(1, self.num_items + 1)]
        supplies = {
            item: np.clip(np.random.normal(mean_supply, supply_std_dev, self.num_warehouses).astype(int), 0, None)
            for item in items
        }
        latitudes, longitudes = self._generate_coordinates(self.num_warehouses)
        warehouses_data = {
            'WarehouseID': [f'Warehouse{i}' for i in range(1, self.num_warehouses + 1)],
            'Latitude': latitudes,
            'Longitude': longitudes,
            **supplies
        }
        return pd.DataFrame(warehouses_data)

    def _generate_random_past_store_records(self, sales_mean, sales_std_dev):
        items = [f'Item{i}' for i in range(1, self.num_items + 1)]
        past_store_records = {
            'StoreID': [f'Store{i}' for i in range(1, self.num_stores + 1)],
            **{
                item: np.clip(np.random.normal(sales_mean, sales_std_dev, self.num_stores).astype(int), 0, None)
                for item in items
            }
        }
        return pd.DataFrame(past_store_records)

    def _generate_random_past_warehouse_records(self, offer_mean, offer_std_dev):
        items = [f'Item{i}' for i in range(1, self.num_items + 1)]
        past_warehouse_records = {
            'WarehouseID': [f'Warehouse{i}' for i in range(1, self.num_warehouses + 1)],
            **{
                item: np.clip(np.random.normal(offer_mean, offer_std_dev, self.num_warehouses).astype(int), 0, None)
                for item in items
            }
        }
        return pd.DataFrame(past_warehouse_records)

class DistanceCalculator:
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # approximate radius of the Earth in km

        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance

    @staticmethod
    def calculate_distance(stores_df, warehouses_df):
        for store_index, store_row in stores_df.iterrows():
            for warehouse_index, warehouse_row in warehouses_df.iterrows():
                distance_km = DistanceCalculator.haversine(store_row['Latitude'], store_row['Longitude'],
                                                          warehouse_row['Latitude'], warehouse_row['Longitude'])
                stores_df.at[store_index, warehouse_row['WarehouseID']] = distance_km

class ReplenishmentSystem:
    def __init__(self, num_items, num_stores, num_warehouses):
        self.num_items = num_items
        self.num_stores = num_stores
        self.num_warehouses = num_warehouses

        self.generator = DataGenerator(num_items, num_stores, num_warehouses)

        self.stores_df = None
        self.warehouses_df = None
        self.past_store_records_df = None
        self.past_warehouse_records_df = None
        self.ranked_warehouses = None

    def generate_data(self, mean_demand, demand_std_dev, max_capacity, mean_supply,
                      supply_std_dev, sales_mean, sales_std_dev, offer_mean, offer_std_dev):
        self.stores_df, self.warehouses_df, self.past_store_records_df, self.past_warehouse_records_df = \
            self.generator.generate_random_data(mean_demand, demand_std_dev, max_capacity,
                                               mean_supply, supply_std_dev, sales_mean,
                                               sales_std_dev, offer_mean, offer_std_dev)

    @staticmethod
    def calculate_warehouse_scores(store_row, warehouses_df, past_warehouse_records_df, weights,
                                   transportation_cost_per_km):
        warehouse_scores = pd.Series(index=warehouses_df.index)

        for warehouse_index, warehouse_row in warehouses_df.iterrows():
            distance_km = store_row[warehouse_row['WarehouseID']]

            warehouse_score = 0
            for item in past_warehouse_records_df.columns[1:]:
                item_demand = store_row[item]
                item_supply = past_warehouse_records_df.loc[warehouse_index, item]
                warehouse_score += item_demand * weights['ItemScore'] + item_supply * weights['DistanceScore']

            transportation_cost = distance_km * transportation_cost_per_km
            warehouse_score -= transportation_cost

            warehouse_scores.at[warehouse_index] = warehouse_score

        return warehouse_scores

    @staticmethod
    def rank_warehouses(warehouse_scores_df):
        ranked_warehouses_df = warehouse_scores_df.rank(axis=1, ascending=False, method='min')
        return ranked_warehouses_df

    def rank_items(self):
        item_columns = [col for col in self.stores_df.columns if 'Item' in col and '_Priority' not in col]

        # Initialize the priority columns
        for item in item_columns:
            self.stores_df[item + '_Priority'] = 0

            # Rank items by priority
        for item in item_columns:
            sorted_stores_df = self.stores_df.sort_values(by=[item], ascending=False)
            for store_index, _ in sorted_stores_df.iterrows():
                self.stores_df.at[store_index, item + '_Priority'] = sorted_stores_df.index.get_loc(store_index) + 1

    def replenishment_by_item(self, max_replenishment):
        replenishment_results = []

        item_columns = [col for col in self.stores_df.columns if 'Item' in col and '_Priority' not in col]

        for item in item_columns:
            sorted_stores_df = self.stores_df.sort_values(by=[item + '_Priority'], ascending=False)

            for store_index, store_row in sorted_stores_df.iterrows():
                required_replenishment = max_replenishment - store_row[item]

                if required_replenishment <= 0:
                    continue

                # Get the ranked warehouses for the specific store
                ranked_warehouses_series = self.ranked_warehouses.loc[store_row['StoreID']]

                # Sort the warehouses by their ranking
                sorted_warehouse_ids = ranked_warehouses_series.sort_values().index

                # Iterate over the sorted warehouse IDs
                for warehouse_id in sorted_warehouse_ids:
                    warehouse_row = self.warehouses_df.loc[self.warehouses_df['WarehouseID'] == warehouse_id].iloc[0]
                    replenish_amount = min(required_replenishment, warehouse_row[item])

                    self.stores_df.at[store_index, item] += replenish_amount
                    self.warehouses_df.loc[self.warehouses_df['WarehouseID'] == warehouse_id, item] -= replenish_amount

                    replenishment_results.append((store_row['StoreID'], item, warehouse_id, replenish_amount))

                    required_replenishment -= replenish_amount

                    if required_replenishment == 0:
                        break

                if required_replenishment > 0:
                    replenishment_results.append((store_row['StoreID'], item, 'Unmet', required_replenishment))

        replenishment_results_df = pd.DataFrame(replenishment_results,
                                                columns=['StoreID', 'Item', 'WarehouseID', 'Replenishment'])

        return replenishment_results_df


    def run_replenishment_system(self, mean_demand, demand_std_dev, max_capacity, mean_supply,
                                 supply_std_dev, sales_mean, sales_std_dev, offer_mean, offer_std_dev, max_replenishment):
        self.generate_data(mean_demand, demand_std_dev, max_capacity, mean_supply, supply_std_dev,
                           sales_mean, sales_std_dev, offer_mean, offer_std_dev)
        print('\nOriginal Stores Data:')
        print(self.stores_df)
        print('\nOriginal Warehouses Data')
        print(self.warehouses_df)

        print("\nPast Store Records (Sales History):")
        print(self.past_store_records_df)
        print("\nPast Warehouse Records (Sent Out History):")
        print(self.past_warehouse_records_df)

        self.stores_df.to_csv('original_stores_data.csv', index=False)
        self.warehouses_df.to_csv('original_warehouses_data.csv', index=False)
        self.past_store_records_df.to_csv('past_store_records.csv', index=False)
        self.past_warehouse_records_df.to_csv('past_warehouse_records.csv', index=False)

        # Fixed line: using DistanceCalculator.calculate_distance
        DistanceCalculator.calculate_distance(self.stores_df, self.warehouses_df)

        self.rank_items()

        warehouse_scores = pd.DataFrame(index=self.stores_df['StoreID'],
                                        columns=self.warehouses_df['WarehouseID'])


        for store_index, store_row in self.stores_df.iterrows():
            # Step 1: Calculate the warehouse scores
            calculated_scores = self.calculate_warehouse_scores(
                store_row,
                self.warehouses_df,
                self.past_warehouse_records_df,
                weights={'ItemScore': 0.3, 'DistanceScore': 0.4},
                transportation_cost_per_km=0.01
            )

            # Ensure the index matches
            calculated_scores.index = warehouse_scores.columns

            # Step 2: Assign the calculated scores to the warehouse_scores DataFrame
            warehouse_scores.loc[store_row['StoreID']] = calculated_scores

        self.ranked_warehouses = self.rank_warehouses(warehouse_scores)

        print('\nWarehouses Scores:')
        print(warehouse_scores)
        print('\nRanked Warehouses:')
        print(self.ranked_warehouses)
        replenishment_results_df = self.replenishment_by_item(max_replenishment)

        return self.stores_df, self.warehouses_df, self.past_store_records_df, self.past_warehouse_records_df, self.ranked_warehouses, replenishment_results_df


if __name__ == '__main__':
    # Constants and example usage
    NUM_ITEMS = 3
    NUM_STORES = 5
    NUM_WAREHOUSES = 3

    MEAN_DEMAND = 100
    DEMAND_STD_DEV = 20

    MAX_CAPACITY = 300
    MEAN_SUPPLY = 300
    SUPPLY_STD_DEV = 50

    SALES_MEAN = 200
    SALES_STD_DEV = 40

    OFFER_MEAN = 1000
    OFFER_STD_DEV = 40

    MAX_REPLENISHMENT = 300

    generator = DataGenerator(NUM_ITEMS, NUM_STORES, NUM_WAREHOUSES)
    replenishment_system = ReplenishmentSystem(NUM_ITEMS, NUM_STORES, NUM_WAREHOUSES)
    stores_df, warehouses_df, past_store_records_df, past_warehouse_records_df, ranked_warehouses, replenishment_results_df = \
        replenishment_system.run_replenishment_system(MEAN_DEMAND, DEMAND_STD_DEV, MAX_CAPACITY,
                                                      MEAN_SUPPLY, SUPPLY_STD_DEV, SALES_MEAN,
                                                      SALES_STD_DEV, OFFER_MEAN, OFFER_STD_DEV,
                                                      MAX_REPLENISHMENT)

    # Display results
    print("\nReplenishment Results:")
    print(replenishment_results_df)

    print("\nFinal Warehouses Data:")
    print(warehouses_df)

    print("\nFinal Stores Data:")
    print(stores_df)

    # Save results to CSV files
    stores_df.to_csv('final-stores_data.csv', index=False)
    warehouses_df.to_csv('final-warehouses_data.csv', index=False)
    replenishment_results_df.to_csv('replenishment_results.csv', index=False)

