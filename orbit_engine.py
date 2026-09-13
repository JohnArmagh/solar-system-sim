import numpy as np

# Orbital Elements (Semi-major axis in AU, Eccentricity, Period in Days, Inclination, etc.)
ORBITS = {
    'Sun':     {'a': 0.0,    'e': 0.0,     'period': 1.0,       'inc': 0.0,   'omega_p': 0.0,    'omega_n': 0.0,    'p_prec': 0.0,       'n_prec': 0.0},
    'Mercury': {'a': 0.3871, 'e': 0.2056,  'period': 87.97,     'inc': 7.00,  'omega_p': 77.46,  'omega_n': 48.33,  'p_prec': 0.004,     'n_prec': -0.002},
    'Venus':   {'a': 0.7233, 'e': 0.0068,  'period': 224.70,    'inc': 3.39,  'omega_p': 131.6,  'omega_n': 76.68,  'p_prec': 0.0002,    'n_prec': -0.001},
    'Earth':   {'a': 1.0000, 'e': 0.0167,  'period': 365.256,   'inc': 0.00,  'omega_p': 102.9,  'omega_n': 0.00,   'p_prec': 0.0003,    'n_prec': 0.0},
    'Moon':    {'a': 0.0800, 'e': 0.0549,  'period': 27.321,    'inc': 5.14,  'omega_p': 0.0,    'omega_n': 0.0,    'p_prec': 360/3232,  'n_prec': -360/6793}
}

ROTATION_PERIODS = {'Sun': 25.05, 'Mercury': 58.65, 'Venus': -243.02, 'Earth': 1.0, 'Moon': 27.321}
RADII = {'Sun': 0.0400, 'Mercury': 0.004, 'Venus': 0.009, 'Earth': 0.010, 'Moon': 0.0027}

def solve_kepler(M, e):
    E = M
    for _ in range(5):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E

def get_orbit_positions(body, t, num_points=200):
    data = ORBITS[body]
    a, e, inc = data['a'], data['e'], np.radians(data['inc'])
    omega_p = np.radians(data['omega_p'] + data['p_prec'] * t)
    omega_n = np.radians(data['omega_n'] + data['n_prec'] * t)
    omega_w = omega_p - omega_n
    
    anomalies = np.linspace(0, 2*np.pi, num_points)
    E = solve_kepler(anomalies, e)
    x_orbit = a * (np.cos(E) - e)
    y_orbit = a * np.sqrt(1.0 - e**2) * np.sin(E)
    
    cos_n, sin_n = np.cos(omega_n), np.sin(omega_n)
    cos_w, sin_w = np.cos(omega_w), np.sin(omega_w)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    
    x_nodes = x_orbit * cos_w - y_orbit * sin_w
    y_nodes = x_orbit * sin_w + y_orbit * cos_w
    
    X = x_nodes * cos_n - y_nodes * sin_n * cos_i
    Y = x_nodes * sin_n + y_nodes * cos_n * cos_i
    Z = y_nodes * sin_i
    return X, Y, Z

def get_body_state(body, t, parent_pos=np.zeros(3)):
    data = ORBITS[body]
    if body == 'Sun':
        return np.zeros(3), {}, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        
    a, e, inc = data['a'], data['e'], np.radians(data['inc'])
    omega_p = np.radians(data['omega_p'] + data['p_prec'] * t)
    omega_n = np.radians(data['omega_n'] + data['n_prec'] * t)
    omega_w = omega_p - omega_n

    M = (2 * np.pi / data['period']) * t
    E = solve_kepler(M, e)
    x_o, y_o = a * (np.cos(E) - e), a * np.sqrt(1.0 - e**2) * np.sin(E)
    
    cos_n, sin_n = np.cos(omega_n), np.sin(omega_n)
    cos_w, sin_w = np.cos(omega_w), np.sin(omega_w)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    
    def transform(x, y):
        xn = x * cos_w - y * sin_w
        yn = x * sin_w + y * cos_w
        return np.array([xn * cos_n - yn * sin_n * cos_i, xn * sin_n + yn * cos_n * cos_i, yn * sin_i]) + parent_pos

    pos = transform(x_o, y_o)
    peri = transform(a * (1.0 - e), 0.0)
    ap = transform(a * (-1.0 - e), 0.0)
    node_asc = transform(a * (np.cos(-omega_w) - e), a * np.sqrt(1.0 - e**2) * np.sin(-omega_w))
    node_desc = transform(a * (np.cos(np.pi - omega_w) - e), a * np.sqrt(1.0 - e**2) * np.sin(np.pi - omega_w))
    
    return pos, {'peri': peri, 'ap': ap, 'node_asc': node_asc, 'node_desc': node_desc}

