import pandas as pd

def network_selecter(carbon_df, networks: list):
    expected_sets = set(['AWS','Azure','GCP'])
    net_set = set(networks)
    extra_nets = net_set - expected_sets
    if extra_nets:
        raise AttributeError(f"Only {expected_sets} are accepted, {extra_nets} is an unaccepted cloud network")
    
    net_zones = pd.read_csv('./Cloud_Network_Zones/network_zone_data.csv').fillna(0)
    net_zones = net_zones[net_zones['ElectricityMaps Zone'] != 0]
    used_nets = net_zones[networks[0]]

    for i in networks[1:]:
        used_nets = used_nets + net_zones[i]
    net_zones['used_nets'] = used_nets
    
    net_zones = net_zones[net_zones['used_nets'] != 0]
    valid_zones = set(net_zones['ElectricityMaps Zone'].tolist())
    our_zones = set(carbon_df.drop(columns=['datetime']).columns.tolist())
    intersect_zone = valid_zones & our_zones

    return carbon_df[['datetime'] + list(intersect_zone)]
