# fdia_logic.py
from test import load_dataset
from test import xyz
def generate_fdia(
    dataset_size: int,
    buses: list[int],
    attack_mag_range: tuple[float, float],
    meter_range: tuple[int, int],
    authentic_name: str,
    fdia_name: str
):
    """
    Runs FDIA generation logic.
    Raise exception if something fails.
    """

    print("Running FDIA with:")
    print(dataset_size, buses, attack_mag_range, meter_range)
    print(authentic_name, fdia_name)
    df=load_dataset(dataset_size)
    stealth_count = xyz(df, buses, attack_mag_range, meter_range, authentic_name, fdia_name)
    

    # Save files, generate dataset, etc.
    return stealth_count

