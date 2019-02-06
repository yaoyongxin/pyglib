# have to be first line
from qiskit import Aer

import numpy
from pyscf import gto, scf, ao2mo
from pyglib.math.matrix_util import unitary_transform_coulomb_matrix

from qiskit_aqua import QuantumInstance
from qiskit_aqua.algorithms import VQE, ExactEigensolver
from qiskit_aqua.components.optimizers import COBYLA
from qiskit_chemistry import FermionicOperator

from qiskit_chemistry.aqua_extensions.components.variational_forms \
        import UCCSD
from qiskit_chemistry.aqua_extensions.components.initial_states \
        import HartreeFock

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
IBMQ.load_accounts()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def computer_hf_integrals(h1e, h2e):
    mol = gto.M()
    n = h1e.shape[0]
    # half filling
    mol.nelectron = n
    # setup hartree-fock calculation
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: numpy.eye(n)
    # get 8-fold permutation symmetry of the integrals
    mf._eri = ao2mo.restore(8, h2e, n)
    mf.kernel()
    # one and two-body integrals with mo
    mohij = mf.mo_coeff.T.dot(h1e).dot(mf.mo_coeff)
    movijkl = unitary_transform_coulomb_matrix(h2e, mf.mo_coeff)
    # adding spin
    ns = n*2
    mohsij = numpy.zeros([ns,ns])
    mohsij[:n,:n] = mohsij[n:,n:] = mohij
    movsijkl = numpy.zeros([ns,ns,ns,ns])
    movsijkl[:n,:n,:n,:n] = movsijkl[:n,:n,n:,n:] = movsijkl[n:,n:,:n,:n] = \
            movsijkl[n:,n:,n:,n:] = 0.5*movijkl
    return mf.mo_coeff, mohsij, movsijkl


def get_qubit_operator(h1, h2=None, qubit_reduction=False, map_type="parity",
        threshold=1.e-10):
    # half-filling
    num_particles = h1.shape[0]/2
    fop = FermionicOperator(h1=h1, h2=h2)
    qop = fop.mapping(map_type=map_type, threshold=threshold)
    if qubit_reduction:
        qop = qop.two_qubit_reduced_operator(num_particles)
    qop.chop(threshold)
    return qop


def check_exact_diag(qop, aux_qops=None):
    exact_eigensolver = ExactEigensolver(qop, k=1, aux_operators=aux_qops)
    ret = exact_eigensolver.run()
    print(ret['aux_ops'])
    print('The computed energy is: {:.12f}'.format(ret['eigvals'][0].real))


def qc_solver(h_qop, num_spin_orbitals, num_particles, map_type, \
        qubit_reduction, aux_qops=None):
    # backends = Aer.backends()
    # backends = IBMQ.backends()
    backend = Aer.get_backend('statevector_simulator')

    # setup COBYLA optimizer
    max_eval = 1000
    cobyla = COBYLA(maxiter=max_eval)
    # setup hartreeFock state
    hf_state = HartreeFock(h_qop.num_qubits, num_spin_orbitals, num_particles,
            map_type, qubit_reduction)
    # setup UCCSD variational form
    var_form = UCCSD(h_qop.num_qubits, depth=1,
            num_orbitals=num_spin_orbitals, num_particles=num_particles,
            active_occupied=[0], active_unoccupied=[0],
            initial_state=hf_state, qubit_mapping=map_type,
            two_qubit_reduction=qubit_reduction, num_time_slices=1)

    # setup VQE
    vqe = VQE(h_qop, var_form, cobyla, operator_mode='matrix', \
            aux_operators=aux_qops)
    quantum_instance = QuantumInstance(backend=backend)
    ret = vqe.run(quantum_instance)
    print(ret['aux_ops'])
    print('The computed ground state energy is: {:.12f}'.format(\
            ret['eigvals'][0]))


def get_dm_qops(mo_coeff, qubit_reduction, map_type):
    n = mo_coeff.shape[0]
    ns = n*2
    qops = []
    for i in range(n):
        for j in range(i+1):
            h1 = numpy.zeros([n, n])
            h1[i,j] = 1.0
            mo_h1 = mo_coeff.T.conj().dot(h1).dot(mo_coeff)
            mo_h1s = numpy.zeros([ns, ns])
            mo_h1s[:n,:n] = mo_h1s[n:,n:] = mo_h1
            qops.append(get_qubit_operator(mo_h1s, \
                    qubit_reduction=qubit_reduction, \
                    map_type=map_type))
    return qops


def qcalc_e_dm(h1e, h2e, check_ed=False, map_type="parity"):
    h1e = numpy.asarray(h1e)
    h2e = numpy.asarray(h2e)
    num_spin_orbitals = h1e.shape[0]*2
    # half-filling
    num_particles = h1e.shape[0]
    if  map_type == "parity":
        qubit_reduction = True
    else:
        qubit_reduction = False
    # get integrals in molecular spin-orbital basis.
    mo_coeff, mo_h1, mo_h2 = computer_hf_integrals(h1e, h2e)
    h_qop = get_qubit_operator(mo_h1, mo_h2, qubit_reduction=qubit_reduction,\
            map_type=map_type)
    # density matrix operator
    aux_qops = get_dm_qops(mo_coeff, qubit_reduction, map_type)
    if check_ed:
        check_exact_diag(h_qop, aux_qops=aux_qops)
    qc_solver(h_qop, num_spin_orbitals, num_particles, map_type,
            qubit_reduction, aux_qops=aux_qops)



if __name__=="__main__":
    n = 2
    h1e = numpy.zeros([n ,n])
    h1e[0,1] = h1e[1,0] = -1.
    h2e = numpy.zeros([n, n, n, n])
    h2e[0,0,0,0] = 2.
    qcalc_e_dm(h1e, h2e, check_ed=True)
