import qiskit as qk
import qiskit_aer as qka
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

# total_error = 100 # %
runtime = 100
iterations = 50
bias = 100
errorvec = np.linspace(0,100,20) # %

# plot preload
xplot = errorvec
results1= np.zeros(len(errorvec))
results2= np.zeros(len(errorvec))

## Defining General Functions

def normalize(x):
    mag = np.sqrt(np.dot(x,np.conjugate(x)))
    v = x/mag
    return v

def normalize_sum(x):
    mag = np.dot(x,np.conjugate(x))
    v = x**2/mag
    return v

def density_to_bloch(rho):
    # Extract the Bloch vector components
    r_x = 2 * np.real(rho[0, 1])
    r_y = 2 * np.imag(rho[0, 1])
    r_z = np.real(rho[0, 0] - rho[1, 1])
    
    # Compute theta and phi.
    # Clip r_z to [-1, 1] to avoid numerical errors causing arccos issues.
    theta = np.arccos(np.clip(r_z, -1, 1))
    phi = np.arctan2(r_y, r_x)
    return theta, phi

def qmccreate(k,params):
        i=k*2+1

        theta = float(params[0])
        phi   = float(params[1])
        lam   = float(params[2])
        
        qc = qk.QuantumCircuit(i, i)
        qc.initialize(initial_state,qubits=0)
        for n in range(0, i-1):
                
            if n%2!=0:

                qc.h(n)
                qc.cx(n,n+1)
    
        for n in range(0, i-1):
            if n%2==0:
                qc.cx(n,n+1)
                qc.h(n)

        for n in range(0, i-1):
            if n%2!=0:
                qc.measure(n,n)
                qc.cx(n,i-1)
            elif n%2==0:
                qc.measure(n,n)
                qc.cz(n,i-1)

        qc.u(theta, phi, lam, i-1)
        qc.save_statevector()
        return qc

    ## Defining Simuation and Fidelity funciton

def qmsimulate(qc):
    # ## DEPENDENT ON DEFINING PAULI MATRICES

    # noise model
    noise_single = qka.noise.mixed_unitary_error(list_single)
    noise_multi = qka.noise.mixed_unitary_error(list_multi)

    noisemodel=qka.noise.NoiseModel()

    noisemodel.add_all_qubit_quantum_error(noise_single,["h"])
    noisemodel.add_all_qubit_quantum_error(noise_multi,["cx","cz"])
    
    # Simulate the circuit
    
    simulator = qka.AerSimulator(method="statevector", noise_model = noisemodel)
    qctrans=qk.transpile(qc,simulator)

    # Results

    result=simulator.run(qctrans).result()


    # Calcuating Fidelity
    final_state = result.get_statevector()
    teleported_state = qk.quantum_info.partial_trace(final_state, range(qc.num_qubits-1))

    fidelity = qk.quantum_info.state_fidelity(teleported_state, initial_state_density)

    return fidelity

# grad

def fidelity_tf(params):
    fid = tf.py_function(
        func=lambda p: qmsimulate(qmccreate(1, p)),
        inp=[params],
        Tout=tf.float64)
    
    return fid

def finite_difference_grad(intialfid,params, delta):
    params_np = params.numpy()
    # Build circuit using the current parameter values:
    base_fid = intialfid
    grads = np.zeros_like(params_np)
    params_up = params_np.copy()
    for i in [0,1,2]:
        params_up[i] += delta
        # Build circuit using the perturbed parameter:
        fid_up = qmsimulate(qmccreate(1, params_up))
        grads[i] = (fid_up - base_fid) / delta
    return grads

def optimize_parameters(intialfid,initial_params, iterations, lr):
    params = tf.Variable(initial_params, dtype=tf.float64)
    for i in tqdm(range(iterations)):
        with tf.GradientTape() as tape:
            # We want to maximize fidelity, so we minimize loss = 1 - fidelity.
            fid = fidelity_tf(params)
            loss = 1 - fid  
        # finite-difference 
        grad_np = finite_difference_grad(intialfid,params,delta = -1e-3) #-4
        grad = tf.convert_to_tensor(grad_np, dtype=tf.float64)
        # Update the parameters (gradient descent step)
        params.assign_sub(lr * grad)
    return params

# def optimize_parameters():
     

# pauli matrices definition

x=np.array([[0,1],[1,0]],dtype=complex)
y=np.array([[0,-1j],[1j,0]],dtype=complex)
z=np.array([[1,0],[0,-1]],dtype=complex)
identity = np.array([[1,0],[0,1]],dtype=complex)

pauli = [identity,x,y,z]

pauli_lower = [identity,x,y,z]

X = np.kron(x,x)
Y = np.kron(y,y)
Z = np.kron(z,z)

Identity = np.kron(identity,identity)

pauli_upper = [Identity,X,Y,Z]

pauli_composite = [0,0,0,0]

alpha = np.random.random() + np.random.random() * 1j
beta = np.random.random() + np.random.random() * 1j

initial_state = np.array([alpha,beta])
initial_state = normalize(initial_state)
initial_state_density = np.outer(initial_state,np.conjugate(initial_state)) 

count = 0
for LIndex in range(len(errorvec)):
##
    total_error = errorvec[LIndex]
    error = total_error /100
    bias =  bias/100

    bias = error * bias
    error = error - bias 


    # weight matrix assignment

    weight = np.ones(13)
    success = 1-error-bias

    weight[0] = success
    weight[1] = bias
    weight[range(2,len(weight))] = error/((len(weight)-2))

    # pauli single matrices

    pauli_single = [0]*len(weight)
    pauli_single[0] = identity

    for i in range(1, len(weight)):
            random_index = np.random.randint(1,len(pauli_lower))
            random_power= np.random.randint(0,2)
            pauli_single[i] = (-1)**random_power * pauli_lower[random_index]

    # pauli composite matrices
    pauli_composite = [0]*len(weight)
    pauli_composite[0] = Identity

    for i in range(1, len(weight)):
            random_index = "np.random.randint(1,len(pauli_upper))"
            random_power= np.random.randint(0,2)
            pauli_composite[i] = (-1)**random_power * np.kron(pauli_lower[eval(random_index)],pauli_lower[eval(random_index)]) 

    # lists
    list_single = [(0,0)]*len(weight)
    list_multi = [(0,0)]*len(weight)

    for i in range(len(weight)):
        list_single[i] = [pauli_single[i],weight[i]]
        list_multi[i] = [pauli_composite[i],weight[i]]
    
    # Circuit Creation

    # Base Circuit
    print("- - - Base Circuit - - -")
    
    sum = 0
    qc=qmccreate(1,[0,0,0])

    for i in tqdm(range(runtime)):
        fidelity = round(qmsimulate(qc),4)
        sum += fidelity
    results1[LIndex] = sum/runtime
    # End
   
    # --- Run the Optimization ---
    print("- - - Optimizing - - -")
    if results1[LIndex] < 8:
        initial_params = [np.arccos(results1[LIndex]), 0, 0]  # initial guesses for theta, phi, lambda
    else:
        initial_params = [0, 0, 0]

    optimized_params = optimize_parameters(results1[LIndex],initial_params, iterations, 1e-4)
    print("Optimized Correction Parameters (theta, phi, lambda):", optimized_params)

    # Corrected Circuit 
    print("- - - Corrected Circuit - - -")
   
    sum = 0
    qc=qmccreate(1,optimized_params)

    for i in tqdm(range(runtime)):
        fidelity = round(qmsimulate(qc),4)
        sum += fidelity
    results2[LIndex] = sum/runtime

    count += 1
    print("Done: " + str(count)+"/"+str(len(errorvec)))

results1 =np.round(results1,2)
results2 =np.round(results2,2)

print(results1,results2)
print(np.sum(results2)-np.sum(results1))


plt.figure()
scatter1=plt.scatter(xplot,results1)
scatter2=plt.scatter(xplot,results2)

scatter1.set_label('Circuit')
scatter2.set_label('Corrected Circuit')

plt.legend(loc='upper right')
plt.xlabel("Error Probability (%)")
plt.ylabel("Avgerage Fidelity")
plt.title("Error Probability versus Avgerage Fidelity for "+str(runtime)+" Trials")
plt.xticks(xplot)
plt.show()