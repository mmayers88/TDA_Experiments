import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import animation


L = 25.0				#linear size
N =	500					#total number of Particles
#N =	100					#total number of Particles(smaller)
rho = N / L **2			#particle density


r0 = 1.0				#interaction Range
deltat = 1.0 			#time steps
factor =0.5			
v0 = r0/deltat*factor	#intial velocity
iterations = 500		#total time steps
eta = 0.1				#delta correlation of white noise

print(" Number of Particles: ",N)
print("Delta correlation of white noise: ",eta)
print("Particle Density: ",rho)
print("Linear Size: ",L)
print("Start Simulation for {}:".format(iterations))



pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)
posArr = np.zeros(pos.shape)

fig, ax= plt.subplots(figsize=(6,6))

qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient[0]), np.sin(orient), orient, clim=[-np.pi, np.pi])
ax.clear()


def init():
	qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient[0]), np.sin(orient), orient, clim=[-np.pi, np.pi])
	global posArr
	posArr = np.stack((posArr,pos))
	return qv,

def animate(i):
	#print(i)

	global orient
	global posArr
	tree = cKDTree(pos,boxsize=[L,L])
	dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')

	#important 3 lines: we evaluate a quantity for every column j
	data = np.exp(orient[dist.col]*1j)
	# construct  a new sparse marix with entries in the same places ij of the dist matrix
	neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
	# and sum along the columns (sum over j)
	S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
	
	
	orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)


	cos, sin= np.cos(orient), np.sin(orient)
	pos[:,0] += cos*v0
	pos[:,1] += sin*v0

	pos[pos>L] -= L
	pos[pos<0] += L
	posArr = np.concatenate((posArr,[pos]))

	qv.set_offsets(pos)
	qv.set_UVC(cos, sin,orient)
	return qv,

#FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=iterations, interval=10, blit=True,repeat=False)
plt.show()
print(posArr.shape)
#np.save("sim_data100.npy",posArr)
