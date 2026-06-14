import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
from math import sqrt

# see Section 3.3 of https://doi.org/10.1002/cnm.2580

# [0, X] \times [0, T], X = dx*(N-1), T = dt*M
N,M = 101, 50
cfl = 0.9
mmHg_to_Pa = 101325./760

def gauss3(f, a, b):
	p = np.array([-sqrt(3./5), 0, sqrt(3./5)])
	p = (b-a)/2 * p + (a+b)/2
	w = np.array([5./9, 8./9, 5./9])
	return (b-a)/2 * (w[0]*f(p[0]) + w[1]*f(p[1]) + w[2]*f(p[2]))

class BaseRP:
	def __init__(self):
		self.Q = np.ndarray(shape=(5,N-1))
		self.rho = 1080.0
		self.mu = 1.0
		self.WB = 0 # well-balanced?
	def IC(self):
		for i in range(N-1):
			xi = (i+0.5)*self.dx # CV center
			self.Q[:,i] = self.RightIC() if xi > self.xg else self.LeftIC()
		self.maxU = np.max(self.WaveSpeed(self.Q))
	def RightIC(self):
		return self.QR
	def LeftIC(self):
		return self.QL
	def CFLTimeStep(self):
		if abs(self.maxU) > 1.0e-10:
			return min(self.dt0, cfl*self.dx/self.maxU)
		return self.dt0
	def phi(self,A,A0):
		return np.power(np.divide(A,A0), self.m) - np.power(np.divide(A,A0), self.n)
	def phiA(self,A,A0):
		return np.divide(self.m*np.power(np.divide(A,A0), self.m-1) - self.n*np.power(np.divide(A,A0), self.n-1), A0)
	def phiA0(self,A,A0):
		return -np.divide(self.m*np.power(np.divide(A,A0), self.m+1) - self.n*np.power(np.divide(A,A0), self.n+1), A)
	def newton(self, phi0, a, a0):
		if False:# and self.n == 0:
			return a0*np.power(phi0+1, 1./self.m)
		else:
			atol, rtol, divtol, res0 = 1.0e-12, 1.0e-8, 1.0e20, 0.0
			iters, maxiters = 0, 100
			a1 = a
			while True:
				res = self.phi(a1,a0)-phi0
				if res0 == 0.0:
					res0 = res
				if abs(res) < atol or abs(res/res0) < rtol:
					#if iters > 0:
					#	print(f"iters {iters} res {res} res0 {res0}")
					return a1
				elif res > divtol or iters > maxiters:
					raise Exception(f"Newton failed to converge res {res} res0 {res0}")
				a1 = a1 - res/self.phiA(a1,a0)
				iters += 1
	def WaveSpeed(self,Q):
		A,_,K,A0,_ = Q
		return np.sqrt(A/self.rho*K*self.phiA(A,A0))
	def Eigvecs(self,Q):
		c = self.WaveSpeed(Q)
		A,q,K,A0,_ = Q
		u = q/A
		phi = self.phi(A,A0)
		phiA0 = self.phiA0(A,A0)
		ArhoU = 1./self.rho * np.divide(A, u*u-c*c)
		r1 = np.matrix([1., u-c, 0, 0, 0,]).T
		r2 = np.matrix([ArhoU * phi, 0., 1., 0., 0.]).T
		r3 = np.matrix([ArhoU * K * phiA0, 0., 0., 1., 0.]).T
		r4 = np.matrix([ArhoU, 0., 0., 0., 1.]).T
		r5 = np.matrix([1., u+c, 0., 0., 0.]).T
		return np.hstack((r1, r2, r3, r4, r5))
	def Lambda(self,Q):
		c = self.WaveSpeed(Q)
		A,q,K,A0,_ = Q
		u = q/A
		return np.diag([u-c, 0., 0., 0., u+c])
	def AbsLambda(self,Q):
		return np.abs(self.Lambda(Q))
	def Mat(self,Q):
		c = self.WaveSpeed(Q)
		A,q,K,A0,_ = Q
		u = q/A
		phi = self.phi(A,A0)
		phiA0 = self.phiA0(A,A0)
		Arho = 1./self.rho * A
		return np.matrix([
			[0, 1., 0., 0., 0.],
			[c*c-u*u, 2*u, Arho*phi, K*Arho*phiA0, Arho],
			[0.,0.,0.,0.,0.],
			[0.,0.,0.,0.,0.],
			[0.,0.,0.,0.,0.]
			])
	def absMat(self,Q):
		R = self.Eigvecs(Q)
		return R*self.AbsLambda(Q)*inv(R)
	def pressure(self,Q):
		A,q,K,A0,pe = Q
		return pe + K*self.phi(A,A0)
	def Psi(self, Qm, Qp, s):
		Q = Qm + s*(Qp-Qm)
		if self.WB:
			A,_,K,A0,pe = Q
			gamma = s*self.pressure(Qp) + (1.-s)*self.pressure(Qm)
			phiS = (gamma-pe)/K
			Q[0] = self.newton(phiS, A, A0)
		return Q
	def dPsi(self, Qm, Qp, s):
		dQ = Qp-Qm
		if self.WB:
			gamma = s*self.pressure(Qp) + (1.-s)*self.pressure(Qm)
			dGamma = self.pressure(Qp)-self.pressure(Qm)
			A,_,K,A0,pe = self.Psi(Qm,Qp,s)
			dA,_,dK,dA0,dpe = dQ
			dPhiS = (dGamma-dpe)/K - dK/K**2*(gamma-pe)
			dQ[0] = (dPhiS - self.phiA0(A,A0)*dA0)/self.phiA(A,A0)
		return dQ
	def ComputeFlux(self,Q):
		D = np.zeros(shape=(5,N,2))
		for i in range(1,N-1):
			Qm,Qp = Q[:,i-1], Q[:,i]
			D[:,i,0] = 0.5*gauss3(lambda s: np.dot(self.Mat(self.Psi(Qm,Qp,s)) + self.absMat(self.Psi(Qm,Qp,s)), self.dPsi(Qm,Qp,s)), 0., 1.)
			D[:,i,1] = 0.5*gauss3(lambda s: np.dot(self.Mat(self.Psi(Qm,Qp,s)) - self.absMat(self.Psi(Qm,Qp,s)), self.dPsi(Qm,Qp,s)), 0., 1.)
		return D
	def ComputeSource(self,Q):
		S = np.zeros(shape=(5,N-1))
		A,q,_,_,_ = Q
		S[1,:] = -self.mu/self.rho * q / np.power(A,2)
		return S
	def step(self):
		self.dt = self.CFLTimeStep()
		D = self.ComputeFlux(self.Q)
		for i in range(N-1):
			self.Q[:,i] = self.Q[:,i] - self.dt/self.dx * (D[:,i+1,1]+D[:,i,0])
		self.maxU = np.max(self.WaveSpeed(self.Q))
	def run(self):
		self.IC()
		t, n = 0.0, 0
		fig,(ax1,ax2) = plt.subplots(2,1)
		x = np.linspace(0,1,N-1)
		while t < self.T-1.0e-10:
			self.step()
			print(f'finished step {n+1} dt {self.dt:.02g} time {t+self.dt:.02g}/{self.T} maxU {self.maxU:.02g}')
			t += self.dt
			n += 1
			if n % 1 == 0:
				alpha = self.Q[0,:] / np.array([self.QL[3] if (i+0.5)*self.dx < self.xg else self.QR[3] for i in range(len(x))])
				U = self.Q[1,:]/self.Q[0,:]
				ax1.set_title('alpha')
				ax2.set_title('U')
				ax1.grid(visible=True), ax2.grid(visible=True)
				ax1.plot(x, alpha)
				ax2.plot(x, U)
				plt.draw()
				plt.pause(0.0001)
				ax1.clear()
				ax2.clear()

class RP1(BaseRP):
	def __init__(self):
		super().__init__()
		self.X = 0.2
		self.xg = 0.5*self.X
		self.T = 0.1
		self.dx = self.X / (N-1)
		self.dt0 = self.T / M
		self.m, self.n = 0.5, 0.0
		self.A0ref = 3.1353e-4
		Kref = 58725.
		self.QL = np.array([0., 0., 1.*Kref , 2.*self.A0ref, 75*mmHg_to_Pa])
		self.QL[0] = self.newton((80*mmHg_to_Pa-self.QL[4])/self.QL[2], self.QL[3], self.QL[3])
		self.QR = np.array([0., 0., 10.*Kref, 1.*self.A0ref, 85*mmHg_to_Pa])
		self.QR[0] = self.newton((80*mmHg_to_Pa-self.QR[4])/self.QR[2], self.QR[3], self.QR[3])

class RP2(BaseRP):
	def __init__(self):
		super().__init__()
		self.X = 0.2
		self.xg = 0.3*self.X
		self.T = 0.007
		self.dx = self.X / (N-1)
		self.dt0 = self.T / M
		self.m, self.n = 0.5, 0.0
		self.A0ref = 3.1353e-4
		Kref = 58725.
		self.QL = np.array([1.6,  1., 1.*Kref , 0.5*self.A0ref, 30*mmHg_to_Pa])
		self.QL[0] *= self.QL[3]
		self.QL[1] *= self.QL[0] # q = A*u
		self.QR = np.array([1.05, 0., 10*Kref, 1.*self.A0ref,  0.])
		self.QR[0] *= self.QR[3]

class RP3(BaseRP):
	def __init__(self):
		super().__init__()
		self.X = 0.2
		self.xg = 0.3*self.X
		self.T = 0.025
		self.dx = self.X / (N-1)
		self.dt0 = self.T / M
		self.m, self.n = 10.0, -1.5
		self.A0ref = 1e-4
		Kref = 5.
		self.QL = np.array([0.9,  0., 1.*Kref , 1.1*self.A0ref, 10*mmHg_to_Pa])
		self.QL[0] *= self.QL[3]
		self.QR = np.array([1.6, 0., 10.*Kref, 1.3*self.A0ref,  5*mmHg_to_Pa])
		self.QR[0] *= self.QR[3]

if __name__ == "__main__":
	RP3().run()