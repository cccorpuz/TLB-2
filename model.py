import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class material:
    def __init__(self,name):
        self.name = name
        self.e_inf, self.e_d, self.tau, self.cond = self.get_material_data(name)
    
    def get_material_data(self, material):
        csv_path = 'C:\\TLB-2\\cole_cole_tissues_gabriel_et_al.csv'
        full_data = pd.read_csv(csv_path,header=None,index_col=0).apply(pd.to_numeric, errors='coerce').to_dict('index')
        try:
            tissue_data_raw = full_data[material]
            tissue_data_raw[3] *= 1e-12
            tissue_data_raw[6] *= 1e-9
            tissue_data_raw[9] *= 1e-6
            tissue_data_raw[12] *= 1e-3
        except KeyError:
            print(f'Material "{material}" not found in database.')

        e_inf = tissue_data_raw[1],
        e_d = tissue_data_raw[2],
        tau = tissue_data_raw[3],
        cond = tissue_data_raw[14]
        return e_inf, e_d, tau, cond

class voxelized_model:
    def __init__(self, input_file, resolution, material_name):
        # TODO: make checks for each parameter input
        self.input_file = input_file
        self.resolution = resolution
        self.material = material(material_name)
        self.model_er, self.model_ed, self.model_tau, self.model_cond = self.load(input_file)

    def load(self, input_file):
        model = cq.importers.importStep(input_file)
        shape = model.val()
        
        # get bounding box of shape
        bb = shape.BoundingBox()

        # grid creation
        # TODO: different grid resolutions?
        x = np.arange(bb.xmin, bb.xmax, self.resolution)
        y = np.arange(bb.ymin, bb.ymax, self.resolution)
        z = np.arange(bb.zmin, bb.zmax, self.resolution)

        nx, ny, nz = len(x), len(y), len(z)
        model_er = np.ones((nx,ny,nz)) # initialize to free space permittivity (e_inf)
        model_ed = np.zeros((nx,ny,nz)) # initialize to free space (0 difference between static e_s and e_inf)
        model_tau = np.ones((nx,ny,nz)) # initialize to free space (doesn't matter value)
        model_cond = np.zeros((nx,ny,nz)) # initialize to free space (no conductivity)

        # TODO: optimize nested loops or distribute the work because of the O(n^3) complexity
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    point = cq.Vector(xi, yj, zk)

                    # Define model shape via material assignment
                    try:
                        if shape.isInside(point):
                            model_er[i,j,k] = self.material.e_inf
                            model_ed[i,j,k] = self.material.e_d
                            model_tau[i,j,k] = self.material.tau
                            model_cond[i,j,k] = self.material.cond
                        else:
                            model_tau[i,j,k] = np.nan
                    except:
                        pass

        return model_er, model_ed, model_tau, model_cond


if __name__ == "__main__":
    head_model_filename = "C:\TLB-2\male_head_2mm - human_body_average_composite.step"
    head_model = voxelized_model(head_model_filename, resolution = 1, material_name = "Muscle")

    plt.plot(head_model.model_er)
    plt.show()