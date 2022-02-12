"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
STOCHASTIC SIMULATION METHODS
FINAL ASSIGNMENT
FEBRUARY 2022

PABLO ROSILLO
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Libraries and dependencies used

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba
import scipy as sc
import graph_tool.all as gta
import networkx as nx
import csv

# Graphs text format similar to LaTex

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rc('font',**{'family':'serif','serif':['Times']})

# Distribution generators

# Each function generates an array with L integers according to a certain 
# distribution. The length of the array must be even and the maximum value
# must not exceed L - 1, as they represent nodes degrees

# Source code for Python generators is available at 
# https://github.com/numpy/numpy/blob/7ccf0e08917d27bc0eba34013c1822b00a66ca6d/numpy/random/mtrand/mtrand.pyx


# All functions try to generate an even array a maximum of repm times

def evendegreelognormal(mu, sigma, L, repm):
    
    # Generates even arrays 50.0654% of the times

    "Generates a log-normal distribution with even sum of degrees."
    "mu is the mean, sigma the std of the inherent Gaussian." 
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around(np.absolute(np.random.lognormal(mu, sigma, L))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg



def evendegreebeta(alpha, beta, maxdeg, L, repm):
    
    # Generates even arrays 49.954% of the times

    "Generates a Beta distribution with even sum of degrees."
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around((maxdeg*np.absolute(np.random.beta(alpha, beta, L)))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreegamma(k, theta, L, repm):
    
    # Generates even arrays 49.9946% of the times

    "Generates a Beta distribution with even sum of degrees."
    "L the size of the degree vector"

    comp = 0; rep = 0;

    while comp == 0 and rep < repm:
        deg = np.around((np.absolute(np.random.gamma(k, theta, L)))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg


def evendegreePoisson(lam, L, repm):
    
    # Generates even arrays 49.9828% of the times

    "Generates a Poisson distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around(np.random.poisson(lam, L)).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreeWeibull(a, lamb, L, repm):
    
    # Generates even arrays 49.9648% of the times

    "Generates a Weibull distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around(lamb*np.random.weibull(a, L)).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg

def evendegreeLomax(a, m, L, repm):
    
    # Generates even arrays 49.9858% of the times

    "Generates a Lomax distribution with even sum of degreeswith exponent lam"

    comp = 0; rep = 0;
    while comp == 0 and rep < repm:
        deg = np.around((m*(np.random.pareto(a, L)+1))).astype(int)
        if (sum(deg) % 2) == 0 and all(i < L for i in deg):
            comp = 1
        rep += 1
    return deg


# SIS simulation function
# We use numba.jit to implement parallelization of the process

@numba.jit(nopython=True, parallel=True)
def sim(L, T, vertices, betarray, mu, nststate, Adj, corr):
    
    meanststate = [numba.int64(x) for x in range(0)]
    errmeanststate = [numba.float64(x) for x in range(0)]
    randininf = np.random.randint(L)
    
    for beta in betarray: # We compute the simulation for each value of beta
        
        ststate = np.zeros(nststate, dtype=numba.int64)
        
        for m in numba.prange(nststate):
    
            htoday = np.zeros(L, dtype=numba.int64) # All susceptible
            htoday[randininf] = 1 # Random node infected
            
            I = 1; 
    
            htomorrow = htoday
            
            for t in numba.prange(T):
                for u in vertices:
                    if htoday[u] == 1:
                        aux = np.delete(vertices, u)
                        for v in aux:
                            if Adj[u][v] == 1 and htoday[v] == 0:
                                if np.random.rand() < beta:
                                    htomorrow[v] = 1
                                    I += 1;
                        if np.random.rand() < mu:
                            htomorrow[u] = 0
                            I -= 1;
                htoday = htomorrow
                
            ststate[m] = I
            
        meanststate.append(np.mean(ststate))
        
        rho = np.zeros(nststate)
        
        if corr == 1: # Computes correlation function
        
        
            z = (ststate-meanststate[-1])/np.std(ststate)
        
            for k in range(nststate):
                for i in range(nststate-k):
                    rho[k] = rho[k] + z[i]*z[i+k]/(nststate-k)
                    
        else:
            
            rho = rho + 1;
        
        errmeanststate.append(np.std(ststate)*np.sqrt(1/nststate))
        
    return meanststate, errmeanststate, rho


# Adjacency matrix creation function following configuration model

def create_Adj(deg, vertices, L):
    
    A = np.zeros((L,L))
    vdist = []
    
    for v in vertices:
        for i in range(int(deg[v])):
            vdist.append(v)
    
    
    while len(vdist) > 0:
        u = np.random.choice(vdist)
        v = np.random.choice(vdist)
        if u != v:
            vdist.remove(v)
            vdist.remove(u)
            A[u][v] = 1
        else:
            if vdist.count(vdist[0]) == len(vdist):
                vdist = []
    
    A = A + A.transpose()
    
    return A, vdist

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Parameters
L = 500

# Mean and standard deviation of a Gaussian to take as reference
mug = 5
stdg = 1

# Lognormal
muln = np.log(mug**2 / np.sqrt(mug**2 + stdg**2))
stdln = np.log(1+stdg**2 / mug**2)

# Beta
kmaxb = 50
alphab = 1
betab = kmaxb/mug - 1

# Gamma
kgamm = 2
thetagamm = mug/kgamm

# Poisson
lambdap = mug

# Weibull
kw = 1
lamdaw = mug

# Lomax
mlomax = 1
alphalomax = 1+1/mug


tic = time.time() # Time counter

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# Graph generation

# Configuration model. Choose distribution (Weibull for example)

deg = evendegreeWeibull(kw, lamdaw, L, 1000)


print('Distribution ready \n')

# Create graph

vertices = np.linspace(0, L-1, L, dtype=int)

Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()

print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


## Barabási-Albert model (undirected)


Adj = np.zeros((L,L))

nmin = 4
ncon = 2

for i in range(nmin):
    for j in range(i+1, nmin):
        Adj[i][j] = 1
Adj = np.maximum(Adj, Adj.transpose()) # Fully connected initial network


for i in range(nmin, L):
    
    deg = np.array([sum(Adj[0])])
    vdist = np.array([0])
    
    for j in range(int(deg[0]-1)):
        vdist = np.append(vdist, 0)
    
    for j in range(1, i):
        deg = np.append(deg, np.array(sum(Adj[j])))
        for k in range(int(deg[j])):
            vdist = np.append(vdist, j)
            
    for j in range(ncon): 
        aux = np.random.choice(vdist)
        if Adj[i][aux] == 0:
            Adj[i][np.random.choice(vdist)] = 1
            Adj[np.random.choice(vdist)][i] = 1
            
deg = np.array([sum(Adj[0])])
vdist = np.array([0])

for j in range(int(deg[0]-1)):
    vdist = np.append(vdist, 0)

for j in range(1, L):
    deg = np.append(deg, np.array(sum(Adj[j])))
    for k in range(int(deg[j])):
        vdist = np.append(vdist, j)
        
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    

## Erdős–Rényi model

# Adjacency matrix

p = 0.01

Adj = np.zeros((L,L))

for i in range(L):
    for j in range(i+1,L):
        if np.random.rand() < p:
            Adj[i][j] = 1

Adj = np.maximum(Adj, Adj.transpose())

deg = np.array(sum(Adj[0]))
for i in range(1, L):
    deg = np.append(deg, sum(Adj[i]))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        


## SIS

T = 1000

betarray = np.linspace(0.01, 0.3, 300) # Infection rate (I + S -> 2I)
mu = 0.5 # Recovery rate (I -> S)
nststate = 20
vertices = np.linspace(0, L-1, L, dtype=int)

corr = 0

meanststate, errmeanststate, rho = sim(L, T, vertices, betarray, mu, nststate, Adj, corr)

toc = time.time()
print(toc-tic, 's elapsed. \n')

plt.figure(1)
plt.errorbar(betarray, meanststate, xerr=None, yerr=errmeanststate,fmt='ok',alpha=0.4)
plt.xlabel(r'$\beta$'); plt.ylabel(r'$I(\infty)$')

if corr == 1:
    
    plt.figure(2)
    plt.plot(rho, '.')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\rho(k)$')


realmeandeg = np.mean(deg)
realmeandeg2 = np.mean(np.power(deg,2))


rows = zip(betarray, meanststate, errmeanststate)

name = f"results/resultsWeibull_L_{L}.txt"

f = open(name, "a+")
f.write(f"#Time elapsed: {toc-tic}\n")
f.write(f"#Real degree mean: {realmeandeg}\n#Real squared degree mean: {realmeandeg2}\n")
f.write(f"#mu: {mu}\n\n#Beta_relmeanststate_relerrmeanststate\n\n")

writer = csv.writer(f)
for row in rows:
    writer.writerow(row)
    
f.close()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# Percolation with all the distributions

nrep = 50

tic = time.time()

fig, axs = plt.subplots(4,2)

## Barabási-Albert model (undirected)
print('BA begins.')

Adj = np.zeros((L,L))

nmin = 4
ncon = 3

for i in range(nmin):
    for j in range(i+1, nmin):
        Adj[i][j] = 1
Adj = np.maximum(Adj, Adj.transpose()) # Fully connected initial network


for i in range(nmin, L):
    
    deg = np.array([sum(Adj[0])])
    vdist = np.array([0])
    
    for j in range(int(deg[0]-1)):
        vdist = np.append(vdist, 0)
    
    for j in range(1, i):
        deg = np.append(deg, np.array(sum(Adj[j])))
        for k in range(int(deg[j])):
            vdist = np.append(vdist, j)
            
    for j in range(ncon): 
        aux = np.random.choice(vdist)
        if Adj[i][aux] == 0:
            Adj[i][np.random.choice(vdist)] = 1
            Adj[np.random.choice(vdist)][i] = 1
            
deg = np.array([sum(Adj[0])])
vdist = np.array([0])

for j in range(int(deg[0]-1)):
    vdist = np.append(vdist, 0)

for j in range(1, L):
    deg = np.append(deg, np.array(sum(Adj[j])))
    for k in range(int(deg[j])):
        vdist = np.append(vdist, j)
        
# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="BA model"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[0,0].plot(x/L, sizes, '.', label=r"$k$", markersize=2)
axs[0,0].plot(x/L, sizes3,'.', label=r"B", markersize=2)
axs[0,0].plot(x/L, sizes5,'.', label=r"C", markersize=2)
axs[0,0].plot(x/L, sizes4,'.', label=r"PR", markersize=2)
axs[0,0].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="R", markersize=2)
axs[0,0].set_ylabel("SLG")
axs[0,0].set_title("BA")
axs[0,0].legend(loc="upper left", framealpha=0.5)



# Beta distribution
print('Beta begins.')

deg = evendegreebeta(alphab, betab, kmaxb, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="Beta dist."):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[0,1].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[0,1].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[0,1].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[0,1].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[0,1].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[0,1].set_title(r"$\beta(k)$")



# ER model
print('ER begins.')

p = 0.01

Adj = np.zeros((L,L))

for i in range(L):
    for j in range(i+1,L):
        if np.random.rand() < p:
            Adj[i][j] = 1

Adj = np.maximum(Adj, Adj.transpose())

deg = np.array(sum(Adj[0]))
for i in range(1, L):
    deg = np.append(deg, sum(Adj[i]))

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="ER model"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[1,0].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[1,0].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[1,0].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[1,0].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[1,0].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[1,0].set_ylabel("SLG")
axs[1,0].set_title("ER")



# Gamma dist
print('Gamma begins.')

deg = evendegreegamma(kgamm, thetagamm, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="Gamma dist"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[1,1].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[1,1].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[1,1].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[1,1].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[1,1].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[1,1].set_title(r"$\Gamma(k)$")



# Log-Normal dist
print('Log-Normal begins.')

deg = evendegreelognormal(muln, stdln, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="LogNormal dist"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[2,0].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[2,0].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[2,0].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[2,0].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[2,0].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[2,0].set_ylabel("SLG")
axs[2,0].set_title("Log-Normal")

# Lomax dist
print('Lomax beigns.')

deg = evendegreeLomax(alphalomax, mlomax, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="Lomax dist"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[2,1].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[2,1].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[2,1].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[2,1].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[2,1].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[2,1].set_title("Lomax")

# Poisson dist
print('Poisson begins.')

deg = evendegreePoisson(lambdap, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="Poisson dist"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[3,0].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[3,0].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[3,0].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[3,0].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[3,0].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[3,0].set_xlabel(r"$\phi_\mathrm{nodes}$")
axs[3,0].set_ylabel("SLG")
axs[3,0].set_title("Poisson")

# Weibull dist
print('Weibull begins.')

deg = evendegreeWeibull(kw, lamdaw, L, 1000)

# Percolation

# Create graph


vertices = np.linspace(0, L-1, L, dtype=int)
Adj, vdist = create_Adj(deg, vertices, L)

toc = time.time()
print('Adjacency matrix ready, ', toc-tic, 's elapsed. \n')

g = gta.Graph()

for i in range(L):
    for j in range(i,L):
        if Adj[i][j] == 1:
            g.add_edge(i,j, add_missing=True)

toc = time.time()
print('Graph ready, ', toc-tic, 's elapsed. \n')

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array


# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, L));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep), desc="Weibull dist"):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,L,1)

axs[3,1].plot(x/L, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
axs[3,1].plot(x/L, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
axs[3,1].plot(x/L, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
axs[3,1].plot(x/L, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
axs[3,1].errorbar(x/L, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
axs[3,1].set_xlabel(r"$\phi_\mathrm{nodes}$")
axs[3,1].set_title("Weibull")











