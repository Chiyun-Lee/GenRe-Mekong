import pandas as pd
from geopy.geocoders import Nominatim

# def load_df():
#     df = pd.read_excel("src/20200705-GenRe-PfMasterData-0.39.xlsx", sheet_name = "GenRe-Mekong")

#     df


def find_location_details(s):
    if "geolocator" not in locals():
        geolocator = Nominatim(user_agent = "MyApp")
    
    location = geolocator.geocode(s, language = "en")

    try:
        return [location.latitude, location.longitude, location.address.split(", ")[-1]]
    
    except AttributeError:
        return [None, None, None]