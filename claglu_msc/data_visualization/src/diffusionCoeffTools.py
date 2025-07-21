from math import isqrt

import numpy as np
import scipy

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import seaborn as sns


# Handle runtime warnings as Errors (used in the Cholesky algorithms)
import warnings
warnings.filterwarnings("error")

# Numpy implementation to compute `Q`
def apply_LDLt_cholesky3x3_numpy(A_inp):
    A = A_inp.copy()
    Lt, Diag, perm = scipy.linalg.ldl(A)
    return Lt[perm, :].dot(np.sqrt(Diag))

# Our Algorithm to compute `Q`
# Reflects what has been implemented in `LangevinHelpers.cpp`
def apply_LDLt_cholesky3x3(A_inp):
    assert A_inp.shape == (3,3)
    A = A_inp.copy()
    row_factors = np.zeros(3)
    D = np.zeros(3)
    
    # Compute first row multiplicators
    row_factors[0] = A[1,0] / A[0,0]
    row_factors[1] = A[2,0] / A[0,0]
    
    # Eliminate value at [1,0]
    A[1,:] = A[1,:] - row_factors[0]*A[0,:]
    
    # Eliminate value at [2,0]
    A[2,:] = A[2,:] - row_factors[1]*A[0,:]
        
    # Eliminate value at [2,1]
    row_factors[2] = A[2,1] / A[1,1]
    A[2,:] = A[2,:] - row_factors[2]*A[1,:]
    
    # Read off values for `D` on Diagonal of `A`
    try:
        D = np.sqrt(np.diag(A))
    except Exception as err:
        print(f'> LDLt-Cholesky: {err}.')
        print(f'> LDLt-Cholesky: Input Matrix:\n{A_inp}.')
        print(f'> LDLt-Cholesky: D:\n{np.diag(A)}')
        raise
    Lt = np.eye(3,3)
    Lt[np.tril_indices(3, k=-1)] = row_factors

    return D*Lt

# Our Algorithm to compute `Q`.
# Takes diagonal values for negative definite matrices for computing `Q`
# Still fails if the diagonal is the culprit for the matrix to be negative definite.
# Has not yet been implemented in `LangevinHelpers.cpp`
def apply_LDLt_cholesky3x3_safe(A_inp):
    assert A_inp.shape == (3,3)
    A = A_inp.copy()
    row_factors = np.zeros(3)
    D = np.zeros(3)
    
    # Compute first row multiplicators
    row_factors[0] = A[1,0] / A[0,0]
    row_factors[1] = A[2,0] / A[0,0]
    
    # Eliminate value at [1,0]
    A[1,:] = A[1,:] - row_factors[0]*A[0,:]
    
    # Eliminate value at [2,0]
    A[2,:] = A[2,:] - row_factors[1]*A[0,:]
        
    # Eliminate value at [2,1]
    row_factors[2] = A[2,1] / A[1,1]
    A[2,:] = A[2,:] - row_factors[2]*A[1,:]
    
    # Read off values for `D` on Diagonal of `A`
    if np.any(np.diag(A) < 0).any(): # Matrix is negative definite
        # Use only diagonal for decomposition
        return apply_LDLt_cholesky3x3(np.diag(np.diag(A_inp)))
    else:
        try:
            D = np.sqrt(np.diag(A))
        except Exception as err:
            print(f'> LDLt-Cholesky: {err}.')
            print(f'> LDLt-Cholesky: Input Matrix:\n{A_inp}.')
            print(f'> LDLt-Cholesky: D:\n{np.diag(A)}')
            raise
        Lt = np.eye(3,3)
        Lt[np.tril_indices(3, k=-1)] = row_factors

    return D*Lt

# Simple Cholesky decomposition (only valid for positive definite matrices)
def apply_simple_cholesky(matrix3x3):
    assert matrix3x3.shape == (3,3)
    L = np.zeros_like(matrix3x3)
    L[0, 0] = np.sqrt(matrix3x3[0, 0]);
    L[1, 0] = matrix3x3[1, 0] / L[0, 0];
    L[1, 1] = np.sqrt(matrix3x3[1, 1] - L[1, 0] * L[1, 0]);
    L[2, 0] = matrix3x3[2, 0] / L[0, 0];
    L[2, 1] = (matrix3x3[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1];
    L[2, 2] = np.sqrt(matrix3x3[2, 2] - L[2, 0] * L[2, 0] - L[2, 1] * L[2, 1]);
    return L

# Extract square numpy matrices from dataframes with individual entries stored as columns.
# If not reducible to a square matrix, it returns a vector
def extract_coeffs(df):
    coeff_values = df.iloc[:,:].to_numpy()
    coeff_size = coeff_values.shape[-1]
    rank1 = isqrt(coeff_size)
    if rank1**2 == coeff_size: # Is an integer sqrt -> return coeff in matrix format
        return coeff_values.reshape(-1,rank1,rank1)
    else: # Has no integer sqrt -> return coeff in vector format
        return coeff_values

# Sort a dataframe according to velocity column
def sort_v_dict(df_dict):
    sorted_df_dict = {}
    for key, df in df_dict.items():
        sorted_df_dict[key] = df.sort_values(by='v')
        
    return sorted_df_dict

# Return Boundary Mask (binary) from given dataframe containing a field where each
# row stores the values at a grid point.
# Assuming nghost == 1.
# Assumes that the dataframe rows correspond to increasing row-major indices of the Matrix-Field.
def generate_boundary_mask(df):
    # Compute cube sidelength
    N3 = df.shape[0]
    N = int((N3+1) ** (1.0/3.0))
    
    # Create index arrays for each dimension
    cube_shape = (N,N,N)
    indices = np.indices(cube_shape)

    # Create a mask for boundary elements
    boundary_mask = np.logical_or.reduce((
        indices[0] == 0,                           # Front face
        indices[0] == cube_shape[0] - 1,           # Back face
        indices[1] == 0,                           # Left face
        indices[1] == cube_shape[1] - 1,           # Right face
        indices[2] == 0,                           # Bottom face
        indices[2] == cube_shape[2] - 1            # Top face
    )).flatten()
    
    return boundary_mask

# Return tuple of disjoint dataframes
# The first entry contains the center cells
# The second entry contains the boundary cells
def split_domains(df):
    boundary_mask = generate_boundary_mask(df)
    return (df.loc[~boundary_mask], df.loc[boundary_mask])

# Averages cells of a dumped field exhibiting the same velocity norm
def extract_avg_values(df_dict):
    avg_dict = {}
    for key, df_dict in coeffs_dict.items():
        Dnorm_avg_dict[key] = df_dict.groupby('v').mean().reset_index()
        
    return avg_dict

# Given cube of size [N^3, M, M] returns [N^2, M, M] at given idx along z axis
def create_cube_zslice(matrix_cube, idx=None):
    N3 = matrix_cube.shape[0]
    trailing_axes_size = matrix_cube.shape[1:]
    N = int((N3+1)**(1.0/3.0))

    if idx is None:
        idx = N//2

    return matrix_cube.reshape(N,N,N,*trailing_axes_size)[:,:,idx,...].reshape(N*N,-1), N

# Give each entry of matrix field a label depending on its definiteness
# Extracts a xy-slice at z=N//2
# Input shape: (N^3,M,M)
# Returns encoded middle slice
def create_encoded_slice(cube_field):
    cube_slice, M = create_cube_zslice(cube_field)
    
    # Create output matrix with scalars as fields (same size as `cube_slice`)
    output_encoding = np.zeros(cube_slice.shape[0], dtype=np.int8)
    
    # Loop through entries
    for idx in range(cube_slice.shape[0]):
        # Compute eigendecomposition
        e_vals, _ = np.linalg.eig(cube_slice[idx].reshape(3,3))
        
        # Assign label according to property
        if np.any(e_vals < 0): # negative
            if np.all(e_vals < 0):
                output_encoding[idx] = 0 # negative definite
            else:
                output_encoding[idx] = 1 # negative semi-definite
        else:
            if np.all(e_vals > 0):
                output_encoding[idx] = 3 # positive definite
            else:
                output_encoding[idx] = 2 # positive semi-definite
    
    return output_encoding.reshape(M,M)

# Plot slice created with `create_encoded_slice()`
def plot_encoded_slice(encoded_data, ax, plot_legend=False, vmax=5e7):
    
    # Create colormap containing a color for each possible value in the data
    colormap = ListedColormap(sns.color_palette("colorblind").as_hex(), N=4)
    cmap_list = [colormap(i) for i in range(colormap.N)]
    # Rotate colormap by one element (results in nicer color combinations for values {1,3})
    cmap_list = cmap_list[1:] + cmap_list[:1]

    labels = ['negative-definite', 'negative semi-definite', 'positive semi-definite', 'positive-definite']

    # Remove labels that are not present in data
    removed_els = 0
    for i in range(len(cmap_list)):
        if np.sum(encoded_data == i) == 0:
            del cmap_list[i - removed_els]
            del labels[i - removed_els]
            removed_els += 1

    # Generate new colormap only containing existing values
    cmap = ListedColormap(cmap_list)

    # Create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=cmap_list[i], label=f'{labels[i]}') for i in range(len(cmap_list)) ]
    
    if plot_legend == True:
        # Put those patched as legend-handles into the legend
        ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize=SizeParams().ticksize)

    
    ax.imshow(encoded_data, cmap=cmap, extent=[-vmax,vmax,-vmax,vmax])
    ax.grid()
    
    return patches