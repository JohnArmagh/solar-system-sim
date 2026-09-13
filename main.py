import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from orbit_engine import get_body_state, get_orbit_positions, RADII, ROTATION_PERIODS
from assets import CONTINENTS, MOON_MARKINGS, project_sphere_features

class SolarSystemSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Orbital Mechanics Simulator")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1c1c1c")
        
        self.time = 0.0
        self.show_orbits = True
        self.show_labels = True
        self.show_panes = True
        self.tracking_target = "Earth"
        
        self.setup_ui()
        self.setup_plots()
        self.animate()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#2d2d2d", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(control_frame, text="SIMULATION CONTROLS", fg="white", bg="#2d2d2d", font=("Arial", 12, "bold")).pack(pady=10)

        tk.Label(control_frame, text="Timelapse Speed (Days/Frame)", fg="white", bg="#2d2d2d").pack()
        self.speed_slider = tk.Scale(control_frame, from_=0.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL, bg="#3d3d3d", fg="white")
        self.speed_slider.set(0.2)
        self.speed_slider.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(control_frame, text="Zoom Window Magnitude Scale", fg="white", bg="#2d2d2d").pack()
        self.zoom_slider = tk.Scale(control_frame, from_=0.005, to=2.5, resolution=0.005, orient=tk.HORIZONTAL, bg="#3d3d3d", fg="white")
        self.zoom_slider.set(0.15)
        self.zoom_slider.pack(fill=tk.X, padx=15, pady=5)

        btn_opts = {"bg": "#4a4a4a", "fg": "white", "activebackground": "#666666", "activeforeground": "white"}
        tk.Button(control_frame, text="Toggle Orbital Paths", command=self.toggle_orbits, **btn_opts).pack(fill=tk.X, padx=20, pady=8)
        tk.Button(control_frame, text="Toggle UI Labels", command=self.toggle_labels, **btn_opts).pack(fill=tk.X, padx=20, pady=8)
        tk.Button(control_frame, text="Toggle Nodal Translucent Panes", command=self.toggle_panes, **btn_opts).pack(fill=tk.X, padx=20, pady=8)

        tk.Label(control_frame, text="Focus Tracking Target Locked:", fg="white", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=15)
        self.track_var = tk.StringVar(value="Earth")
        for target in ["Sun", "Mercury", "Venus", "Earth", "Moon"]:
            tk.Radiobutton(control_frame, text=f"Focus on {target}", variable=self.track_var, value=target, 
                           command=self.update_tracking, bg="#2d2d2d", fg="white", selectcolor="#1c1c1c").pack(anchor=tk.W, padx=30, pady=3)

    def setup_plots(self):
        self.fig, (self.ax_top, self.ax_side) = plt.subplots(2, 1, figsize=(10, 9))
        self.fig.patch.set_facecolor('black')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def toggle_orbits(self): self.show_orbits = not self.show_orbits
    def toggle_labels(self): self.show_labels = not self.show_labels
    def toggle_panes(self): self.show_panes = not self.show_panes
    def update_tracking(self): self.tracking_target = self.track_var.get()

    def animate(self):
        self.time += self.speed_slider.get()
        self.ax_top.clear()
        self.ax_side.clear()
        
        for ax in [self.ax_top, self.ax_side]:
            ax.set_facecolor('black')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        pos, elems = {}, {}
        pos['Sun'], elems['Sun'] = get_body_state('Sun', self.time)
        pos['Mercury'], elems['Mercury'] = get_body_state('Mercury', self.time, pos['Sun'])
        pos['Venus'], elems['Venus'] = get_body_state('Venus', self.time, pos['Sun'])
        pos['Earth'], elems['Earth'] = get_body_state('Earth', self.time, pos['Sun'])
        pos['Moon'], elems['Moon'] = get_body_state('Moon', self.time, pos['Earth'])

        center_offset = np.copy(pos[self.tracking_target])

        if self.show_panes:
            for body in ['Mercury', 'Venus', 'Moon']:
                el = elems[body]
                n1, n2 = el['node_asc'] - center_offset, el['node_desc'] - center_offset
                self.ax_top.plot([n1[0], n2[0]], [n1[1], n2[1]], color='teal', alpha=0.3, linestyle='--')
                self.ax_side.fill_between([n1[0], n2[0]], [n1[2], n2[2]], color='teal', alpha=0.15)

        if self.show_orbits:
            for body in ['Mercury', 'Venus', 'Earth', 'Moon']:
                p_base = pos['Earth'] if body == 'Moon' else pos['Sun']
                X, Y, Z = get_orbit_positions(body, self.time, num_points=250)
                color = 'cyan' if body == 'Moon' else ('blue' if body == 'Earth' else 'gray')
                self.ax_top.plot(X + p_base[0] - center_offset[0], Y + p_base[1] - center_offset[1], color=color, alpha=0.4, lw=1)
                self.ax_side.plot(X + p_base[0] - center_offset[0], Z + p_base[2] - center_offset[2], color=color, alpha=0.4, lw=1)

        for body in ['Mercury', 'Venus', 'Earth', 'Moon']:
            el = elems[body]
            for key, pt in el.items():
                pt_shifted = pt - center_offset
                if self.show_orbits:
                    self.ax_top.scatter(pt_shifted[0], pt_shifted[1], color='gold' if body=='Moon' else 'white', s=8, alpha=0.6)
                    self.ax_side.scatter(pt_shifted[0], pt_shifted[2], color='gold' if body=='Moon' else 'white', s=8, alpha=0.6)
                if self.show_labels:
                    self.ax_top.text(pt_shifted[0], pt_shifted[1], f"{body} {key.upper()}", color='yellow', fontsize=7, alpha=0.7)
                    self.ax_side.text(pt_shifted[0], pt_shifted[2], f"{body} {key.upper()}", color='yellow', fontsize=7, alpha=0.7)

        for body, position in pos.items():
            pt_shifted = position - center_offset
            r = RADII[body]
            rot_angle = (2 * np.pi / ROTATION_PERIODS[body]) * self.time
            
            if body == 'Earth':
                l_top, l_side = project_sphere_features(CONTINENTS, r, pt_shifted, rot_angle)
                for lx, ly in l_top: self.ax_top.plot(lx, ly, color='green', lw=1)
                for lx, lz in l_side: self.ax_side.plot(lx, lz, color='green', lw=1)
            elif body == 'Moon':
                l_top, l_side = project_sphere_features(MOON_MARKINGS, r, pt_shifted, rot_angle)
                for lx, ly in l_top: self.ax_top.plot(lx, ly, color='darkgray', lw=0.7)
                for lx, lz in l_side: self.ax_side.plot(lx, lz, color='darkgray', lw=0.7)
            else:
                color_map = {'Sun': 'orange', 'Mercury': 'brown', 'Venus': 'gold'}
                self.ax_top.add_patch(plt.Circle((pt_shifted[0], pt_shifted[1]), r, color=color_map[body]))
                self.ax_side.add_patch(plt.Circle((pt_shifted[0], pt_shifted[2]), r, color=color_map[body]))

            if r / self.zoom_slider.get() < 0.01:
                self.ax_top.scatter(pt_shifted[0], pt_shifted[1], color='cyan' if body=='Earth' else 'white', s=15)
                self.ax_side.scatter(pt_shifted[0], pt_shifted[2], color='cyan' if body=='Earth' else 'white', s=15)

            if self.show_labels:
                self.ax_top.text(pt_shifted[0] + r, pt_shifted[1] + r, body, color='white', fontsize=9)
                self.ax_side.text(pt_shifted[0] + r, pt_shifted[2] + r, body, color='white', fontsize=9)

        zoom_val = self.zoom_slider.get()
        self.ax_top.set_xlim(-zoom_val, zoom_val)
        self.ax_top.set_ylim(-zoom_val, zoom_val)
        self.ax_side.set_xlim(-zoom_val, zoom_val)
        self.ax_side.set_ylim(-zoom_val / 2, zoom_val / 2)

        self.ax_top.set_title("TOP-DOWN VIEW (XY PLANE Projection)", color='white', fontsize=10)
        self.ax_side.set_title("SIDE VIEW (XZ PLANE Projection)", color='white', fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()
        self.root.after(20, self.animate)

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarSystemSimulator(root)
    root.mainloop()

