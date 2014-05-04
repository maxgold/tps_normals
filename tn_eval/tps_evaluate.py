import tps_utils


def tps_eval(x_na, y_ng, nwsize=0.02, delta=0.05):
    """
    delta: edge length
    
    """
    
    
    e_x = tps_utils.find_all_normals_naive(x_na, nwsize, flip_away=True, project_lower_dim=True)
    e_y = tps_utils.find_all_normals_naive(y_ng, nwsize, flip_away=True, project_lower_dim=True)
    
    
    