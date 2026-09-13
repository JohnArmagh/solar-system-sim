import numpy as np

def gen_sphere_features(num_features=6, seed=42):
    np.random.seed(seed)
    features = []
    for _ in range(num_features):
        lon = np.linspace(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), 20)
        lat = np.sin(lon * np.random.randint(1, 4)) * np.random.uniform(0.2, 0.8)
        features.append((lon, lat))
    return features

CONTINENTS = gen_sphere_features(7, seed=42)
MOON_MARKINGS = gen_sphere_features(5, seed=24)

def project_sphere_features(features, radius, center, rot_angle):
    lines_top_down, lines_side = [], []
    for lon, lat in features:
        r_lon = lon + rot_angle
        x = radius * np.cos(lat) * np.sin(r_lon)
        y = radius * np.cos(lat) * np.cos(r_lon)
        z = radius * np.sin(lat)
        
        visible_mask = y >= 0 
        if np.any(visible_mask):
            lines_top_down.append((x[visible_mask] + center[0], y[visible_mask] + center[1]))
            lines_side.append((x[visible_mask] + center[0], z[visible_mask] + center[2]))
            
    return lines_top_down, lines_side

