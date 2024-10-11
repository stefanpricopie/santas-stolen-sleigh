import pandas as pd
import numpy as np


class Solution:
    def __init__(self, solution_input, gifts_file_path=None):
        """
        Initialize the solution object with either:
        - a CSV file (for Kaggle submission),
        - or a vector (list or numpy array) representing the solution.

        This is divided into two parts: loading the problem data and loading/validating the solution.

        :param solution_input: Path to the solution CSV file or a vector of Sleigh assignments.
        :param gifts_file_path: Path to the gifts CSV file (optional, for main problem).
        """
        # Load the problem data (gifts)
        self._get_problem(gifts_file_path)

        # Load and validate the solution (CSV or vector)
        self._get_solution(solution_input)

    def _get_problem(self, gifts_file_path):
        """
        Load the gift data, including GiftId, Latitude, Longitude, and Weight.
        :param gifts_file_path: Path to the gifts file or a DataFrame.
        """
        # Optionally load the gifts file for the main problem
        if isinstance(gifts_file_path, str) and gifts_file_path.endswith('.csv'):
            # Load the gifts CSV file
            self.gifts_df = pd.read_csv(gifts_file_path)
        elif isinstance(gifts_file_path, pd.DataFrame):
            # Use the DataFrame directly
            self.gifts_df = gifts_file_path
        else:
            # Default to the main problem's gifts.csv
            self.gifts_df = pd.read_csv("data/gifts.csv")  # Load the default gifts.csv file

        # # Extract gift data columns
        # self.gift_id = self.gifts_df['GiftId'].values    # Gifts are 1-indexed
        # self.lat = self.gifts_df['Latitude'].values
        # self.lon = self.gifts_df['Longitude'].values
        # self.weight = self.gifts_df['Weight'].values

    def _get_solution(self, solution_input):
        """
        Load the solution either from a CSV file or from a vector.
        Also validates the solution during this process.

        :param solution_input: Path to the solution CSV file or a vector of Sleigh assignments.
        """
        if isinstance(solution_input, str) and solution_input.endswith('.csv'):
            # Case 1: solution_input is a CSV file (Kaggle submission)
            # Order of gift ids is different from gifts_df
            solution_pairs = pd.read_csv(solution_input)
            assert sorted(solution_pairs.columns.tolist()) == ['GiftId', 'TripId']

            self.solution_df = pd.merge(self.gifts_df, solution_pairs, how='right', on='GiftId')
        elif isinstance(solution_input, (list, np.ndarray)):
            # Case 2: solution_input is a vector of Sleigh assignments
            raise NotImplementedError
        else:
            raise ValueError("solution_input must be either a CSV file path or a vector (list/array)")

    def calculate_wrw(self):
        """
        Calculate the final value based on the vector or CSV solution.
        For the main problem, use gifts.csv if available.
        """
        return total_wrw(self.solution_df)

    def to_kaggle(self, output_path):
        """
        Convert the vector solution back to a CSV format for Kaggle submission.
        If the solution was initialized with a vector, this will generate the CSV.
        """
        # Save the DataFrame as a CSV
        self.solution_df[["GiftId","TripId"]].to_csv(output_path, index=False)
        print(f"Solution saved to {output_path}")


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2.0 * 6371.0

    return c * np.arcsin(np.sqrt(a))

def trip_wrw(one_trip):    # one_trip is array

    # Create arrays of the longitudes and latitudes
    lon1 = one_trip[:,2]
    lon2 = one_trip[:,2]
    lat1 = one_trip[:,1]
    lat2 = one_trip[:,1]

    # Add the north pole at the start and end of the journey
    lon1 = np.concatenate([[0], lon1])
    lon2 = np.concatenate([lon2, [0]])
    lat1 = np.concatenate([[90], lat1])
    lat2 = np.concatenate([lat2, [90]])

    # Calculate the distances between all gifts
    tripdistances = haversine_np(lon1, lat1, lon2, lat2)

    # Create array of weights
    weight = one_trip[:,3]
    weight = np.flip(weight)
    weight = np.cumsum(weight)
    weight = np.flip(weight)

    # Add cart weight
    weight = weight + 10

    # Ensure the only weight on the way back is the cart
    weight = np.concatenate([weight, [10]])

    # Calculate the weighted distances
    distances = tripdistances * weight
    totaldist = np.sum(distances)

    return totaldist

def total_wrw(solution_df):   # all_trips is DataFrame
    # returns total WRW, maximum weight and checks if GiftIDs and TripIDs are respected
    # turn into array
    all_gifts = np.array(solution_df)

    all_trips_array = [all_gifts[all_gifts[:,4]==k] for k in np.unique(all_gifts[:,4])]
    wrw = 0
    for trip in all_trips_array:
        wrw += trip_wrw(trip)

    return wrw