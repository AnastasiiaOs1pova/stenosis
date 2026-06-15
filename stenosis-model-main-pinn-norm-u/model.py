import numpy as np
import json
from math import sqrt, pi, sin, exp, floor
from scipy.linalg import lu_factor, lu_solve
import time
from multiprocessing.dummy import Pool as ThreadPool
from flux_helper import flux_helper
import matplotlib.pyplot as plt
from pinn import get_pinn


params_map = {
	"T_FINAL": "total_time",
	"DX": "space_step",
	"DT": "time_step",
	"RHO": "density",
	"MU": "viscosity",
	"ALASTRUEY_GAMMA": "alastruey_gamma",
	"SCHEME_GAMMA": "scheme_gamma",
	"L": "length",
	"DIAM": "diameter",
	#"DIAM_OUT": "diameter_out",
	"AREA_DIA": "area_diastolic",
	"P_DIA": "pressure_diastolic",
	"C_SPEED": "perturbation_speed",
	"H_WALL": "wall_thickness",
	"E_YOUNG": "young_modulus",
	"T_PER": "time_period",
	"BC_TYPE": "bc_type",
	"BC_OUT": "outlet_type",
	"V_ID": "vessel_index",
	"P_OUT": "outflow_pressure",
	"WK_R1": "characteristic_impedance",
	"WK_R2": "total_peripheral_resistance",
	"WK_C": "total_arterial_compliance",
	"FLUX_TYPE": "flux_type",
	"FLUX_FILENAME": "flux_filename",
	"FLUX_ID": "analytics_index",
	"STATS_ID": "stats_index",
	"C_INLET": "inlet_concentration",
	"USE_C": "compute_concentration"
}

class DictWithAttributeAccess(dict):
	def __getattr__(self, key):
		return self[params_map[key]] if key in params_map and params_map[key] in self else self[key]

	def __setattr__(self, key, value):
		if key in params_map:
			self[params_map[key]] = DictWithAttributeAccess(value) if type(value) is dict else value
		else:
			self[key] = value

class Vessel(DictWithAttributeAccess):
	def __init__(self, data, i):
		self.C_SPEED = 0.0
		self.AREA_DIA = 0.0
		DictWithAttributeAccess.__init__(self, data['vessel'][i])
		self.ALASTRUEY_BETA = 4./3 * sqrt(pi) * self.E_YOUNG * self.H_WALL
		self.tinn = np.zeros(5)

	def post_setup(self, model):
		self.bc_types = ['','']
		self.ALASTRUEY_GAMMA = model.ALASTRUEY_GAMMA
		self.SCHEME_GAMMA = model.SCHEME_GAMMA
		self.bc_order = model.bc_order
		self.RHO = model.RHO
		self.MU = model.MU
		self.DX = model.DX
		self.DT = min(model.DT, self.CFLTimeStep(model.CFL))
		if self.DT < model.DT:
			model.DT = self.DT
		self.N = floor( self.L / self.DX) + 1
		self.AREA_IN = pi*self.DIAM**2 / 4
		#self.AREA_OUT = pi*self.DIAM_OUT**2 / 4
		if self.AREA_DIA == 0.0:
			self.AREA_DIA = self.AREA_IN
		self.S = np.ones(self.N) * self.AREA_IN
		self.U = np.zeros(self.N)
		self.P = np.ones(self.N) * self.P_DIA
		self.FS = np.zeros(self.N)
		self.FU = np.zeros(self.N)
		self.W1S = np.zeros(self.N)
		self.W1U = np.zeros(self.N)
		self.USE_C = model.USE_C
		if model.USE_C:
			self.C = np.zeros(self.N)

	def linear_area(self, ind):
		return self.AREA_DIA
		x = ind*1.0/self.L * self.DX
		diam = (1.-x/L)*DIAM + x/L * DIAM_OUT
		return pi*diam**2/4

	def get_area(self, pressure):
		return np.power(np.sqrt(self.AREA_DIA) + self.AREA_DIA/self.ALASTRUEY_BETA*(pressure - self.P_DIA), 2)

	def pressure(self, area):
		return self.P_DIA + self.ALASTRUEY_BETA/self.AREA_DIA*(np.sqrt(area) - sqrt(self.AREA_DIA))

	def pressure_deriv(self, area):
		return self.ALASTRUEY_BETA/(2*self.AREA_DIA) * 1.0 / np.sqrt(area)

	def pressure_deriv2(self, area):
		return -self.ALASTRUEY_BETA/(4*self.AREA_DIA) * 1.0 / (area * np.sqrt(area) )

	def perturbation_velocity_coef(self, area):
		return np.sqrt(1.0 / (self.RHO * area) * self.pressure_deriv(area) )

	def friction(self, area, velocity):
		return -2.0*pi*(2.0+self.ALASTRUEY_GAMMA)*self.MU / self.RHO * np.divide(velocity,area)

	def CFLTimeStep(self, CFL):
		maxvel = 0.0
		if self.C_SPEED > 0.0:
			maxvel = self.C_SPEED
		else:
			maxvel = sqrt(self.E_YOUNG/self.RHO)
		assert maxvel > 0.0, 'Fail to setup CFL time step, incorrect input parameters'
		return CFL*self.DX/maxvel

	def step_inner_points(self):
		t = np.zeros(shape=len(self.tinn)+1)
		t[0] = time.time()
		tau, h = self.DT, self.DX
		N = self.N
		SCHEME_GAMMA = self.SCHEME_GAMMA
		self.FS = np.multiply(self.S,self.U)
		self.FU = np.power(self.U,2)/2 + self.pressure(self.S) / self.RHO
		t[1] = time.time()
		# prepare work array
		self.W1S[1:-1] = self.S[1:-1] - 0.5*tau/h * (self.FS[2:] - self.FS[:-2])
		self.W1U[1:-1] = self.U[1:-1] - 0.5*tau/h * (self.FU[2:] - self.FU[:-2])
		if self.bc_order == 1:
			self.W1S[0]  = self.S[0]  - tau/h * (self.FS[1] - self.FS[0])
			self.W1U[0]  = self.U[0]  - tau/h * (self.FU[1] - self.FU[0])
			self.W1S[-1] = self.S[-1] - tau/h * (self.FS[-1] - self.FS[-2])
			self.W1U[-1] = self.U[-1] - tau/h * (self.FU[-1] - self.FU[-2])
		else:
			self.W1S[0]  = self.S[0]  - 0.5*tau/h * (3*self.FS[2] - 4*self.FS[1] + self.FS[0])
			self.W1U[0]  = self.U[0]  - 0.5*tau/h * (3*self.FU[2] - 4*self.FU[1] + self.FU[0])
			self.W1S[-1] = self.S[-1] + 0.5*tau/h * (3*self.FS[-3] - 4*self.FS[-2] + self.FS[-1])
			self.W1U[-1] = self.U[-1] + 0.5*tau/h * (3*self.FU[-3] - 4*self.FU[-2] + self.FU[-1])
		t[2] = time.time()
		# don't slice reused arrays if it can be cached:
		Sp,S0,Sm = self.S[2:],self.S[1:-1],self.S[:-2]
		Up,U0,Um = self.U[2:],self.U[1:-1],self.U[:-2]
		# compute next time step values into work array
		Sp12,Up12 = 0.5*(S0+Sp),0.5*(U0+Up)
		Sm12,Um12 = 0.5*(S0+Sm),0.5*(U0+Um)
		v_plus  = self.perturbation_velocity_coef(Sp12)
		v_minus = self.perturbation_velocity_coef(Sm12)
		sigma_plus  = tau/h * np.vstack([Up12 - v_plus*Sp12, Up12 + v_plus*Sp12])
		sigma_minus = tau/h * np.vstack([Um12 - v_minus*Sm12, Um12 + v_minus*Sm12])
		t[3] = time.time()
		# assert np.all(sigma_plus[0] < 0.) and np.all(sigma_minus[0] < 0.), 'invalid sign in characteristic'
		# assert np.all(sigma_plus[1] > 0.) and np.all(sigma_minus[1] > 0.), 'invalid sign in characteristic'
		# assert np.all(abs(sigma_plus) < 1.), 'possibly too big time step, CFL violation'
		# assert np.all(abs(sigma_minus) < 1.), 'possibly too big time step, CFL violation'
		b_plus  = 0.5*np.multiply(np.abs(sigma_plus),  1. + 5./19*(1.-SCHEME_GAMMA)*(1.-np.abs(sigma_plus )) )
		b_minus = 0.5*np.multiply(np.abs(sigma_minus), 1. + 5./19*(1.-SCHEME_GAMMA)*(1.-np.abs(sigma_minus)) )
		d_plus  = 6./19 * (1.-SCHEME_GAMMA) * np.multiply(sigma_plus,  (1./np.abs(sigma_plus)  - 1.))
		d_minus = 6./19 * (1.-SCHEME_GAMMA) * np.multiply(sigma_minus, (1./np.abs(sigma_minus) - 1.))
		t[4] = time.time()
		def mult_OMEGA_INV_C_OMEGA_V2(vk,c,area,vel):
			return np.vstack([\
				0.5*np.multiply(c[0]+c[1],area) + 0.5*np.multiply(np.divide(c[1]-c[0], vk), vel),\
				0.5*np.multiply(np.multiply(c[1]-c[0],vk),area) + 0.5*np.multiply(c[1]+c[0],vel) ])
		dWb1 = mult_OMEGA_INV_C_OMEGA_V2(v_plus, b_plus, \
			Sp - S0, Up - U0)
		dWb0 = mult_OMEGA_INV_C_OMEGA_V2(v_minus, b_minus, \
			S0 - Sm, U0 - Um)
		dWd1 = mult_OMEGA_INV_C_OMEGA_V2(v_plus, d_plus, \
			self.W1S[2:] - Sp + self.W1S[1:-1] - S0, \
			self.W1U[2:] - Up + self.W1U[1:-1] - U0)
		dWd0 = mult_OMEGA_INV_C_OMEGA_V2(v_minus, d_minus, \
			self.W1S[:-2] - Sm + self.W1S[1:-1] - S0, \
			self.W1U[:-2] - Um + self.W1U[1:-1] - U0)
		FRIC = tau*self.friction(S0, U0)
		self.S[1:-1] = self.W1S[1:-1] + dWb1[0] - dWb0[0] + dWd1[0] - dWd0[0]
		self.U[1:-1] = self.W1U[1:-1] + dWb1[1] - dWb0[1] + dWd1[1] - dWd0[1] + FRIC
		t[5] = time.time()
		for i in range(len(self.tinn)):
			self.tinn[i] += t[i+1]-t[i]
		if self.USE_C:
			self.C[1:-1] -= tau/h*self.U[1:-1] * np.where(self.U[1:-1] > 0, self.C[1:-1]-self.C[:-2], self.C[2:] - self.C[1:-1])

class BC(DictWithAttributeAccess):
	def __init__(self, data):
		DictWithAttributeAccess.__init__(self, data)
		# pairs (v_id, bc_side)
		# bc_side =  'left' -- model.vessels[v_id].A[0] corresponds to this BC
		# bc_side = 'right' -- model.vessels[v_id].A[-1] corresponds to this BC
		# bc_side =	  '' -- unset, should be a bug or error in vessel connectivity
		if self.BC_TYPE in ('junction', 'stenosis_pinn'):
			self.indices = [[int(s),''] for s in data['vessel_index'].split(';')]

	def post_setup(self, model):
		if self.BC_TYPE == 'inlet':
			self.indices = [[self.V_ID,'left']]
			model.vessels[self.V_ID].bc_types[0] = 'inlet'
			if self.FLUX_TYPE == 'file':
				self.FH = flux_helper(self.FLUX_FILENAME)
		elif self.BC_TYPE == 'outlet':
			self.indices = [[self.V_ID,'']]


	def compute_bc_coefs(self, model, ind):
		v_id, bc_side = self.indices[ind]
		vessel = model.vessels[v_id]
		if bc_side == 'left':
			i0, di, sign = 0, 1, 1
		else:
			i0, di, sign = vessel.N-1, -1, -1
		s, u = [0.]*3, [0.]*3
		for i in range(3):
			s[i] = vessel.S[i0 + i*di]
			u[i] = vessel.U[i0 + i*di]
		if model.bc_order == 1:
			se, ue, coef = s[1], u[1], 1.0
		else:
			se, ue, coef = 2*s[1] - 0.5*s[2], 2*u[1] - 0.5*u[2], 1.5
		w = vessel.perturbation_velocity_coef(s[0])
		sigma = vessel.DT/vessel.DX * (u[0] - sign*w*s[0])

		alpha = sign*w
		beta = (w*(sigma*se - sign*s[0]) + (u[0] - sign*sigma*ue) - sign*vessel.DT*vessel.friction(s[0], u[0])) \
			/ (1.-sign*coef*sigma)
		return alpha, beta

	# WINDKESSEL BC
	def wk_newton_f(self, vessel, dt, s, s0, u0, a, b):
		return 1.0/dt * (
			s*(a*s+b) * ( dt*(self.WK_R1+self.WK_R2) + self.WK_C*self.WK_R2*self.WK_R1 )
			- dt*(vessel.pressure(s) - self.P_OUT)
			- self.WK_C*self.WK_R2*vessel.pressure_deriv(s) * (s - s0)
			- self.WK_C*self.WK_R2*self.WK_R1 * (s0*u0)
			)

	def wk_newton_dfds(self, vessel, dt, s, s0, u0, a, b):
		return  1.0/dt * (
			(2*a*s+b) * ( dt*(self.WK_R1+self.WK_R2) + self.WK_C*self.WK_R2*self.WK_R1 )
			- dt*vessel.pressure_deriv(s)
			- self.WK_C*self.WK_R2*( vessel.pressure_deriv2(s) * (s - s0) + vessel.pressure_deriv(s) )
			)

	def analytic_flux(self, t):
		if self.FLUX_ID == 0:
			return exp(-10000.0 * (t-0.05)*(t-0.05))
		elif self.FLUX_ID == 1:
			# flux in ml/s:
			T = self.T_PER
			return 10e5*(7.9853e-06\
				+2.6617e-05*sin(2*pi*t/T+0.29498)+2.3616e-05*sin(4*pi*t/T-1.1403)-1.9016e-05*sin(6*pi*t/T+0.40435)\
				-8.5899e-06*sin(8*pi*t/T-1.1892)-2.436e-06*sin(10*pi*t/T-1.4918)+1.4905e-06*sin(12*pi*t/T+1.0536)\
				+1.3581e-06*sin(14*pi*t/T-0.47666)-6.3031e-07*sin(16*pi*t/T+0.93768)-4.5335e-07*sin(18*pi*t/T-0.79472)\
				-4.5184e-07*sin(20*pi*t/T-1.4095)-5.6583e-07*sin(22*pi*t/T-1.3629)+4.9522e-07*sin(24*pi*t/T+0.52495)\
				+1.3049e-07*sin(26*pi*t/T-0.97261)-4.1072e-08*sin(28*pi*t/T-0.15685)-2.4182e-07*sin(30*pi*t/T-1.4052)\
				-6.6217e-08*sin(32*pi*t/T-1.3785)-1.5511e-07*sin(34*pi*t/T-1.2927)+2.2149e-07*sin(36*pi*t/T+0.68178)\
				+6.7621e-08*sin(38*pi*t/T-0.98825)+1.0973e-07*sin(40*pi*t/T+1.4327)-2.5559e-08*sin(42*pi*t/T-1.2372)\
				-3.5079e-08*sin(44*pi*t/T+0.2328))
		elif self.FLUX_ID == 2:
			T = self.T_PER
			return 1e6*(
				3.1199+7.7982*sin(2*pi*t/T+0.5769)+4.1228*sin(4*pi*t/T-0.8738)-1.0611*sin(6*pi*t/T+0.7240)+0.7605*sin(8*pi*t/T-0.6387)
				-0.9148*sin(10*pi*t/T+1.1598)+0.4924*sin(12*pi*t/T-1.0905)-0.5580*sin(14*pi*t/T+1.042)+0.3280*sin(16*pi*t/T-0.5570)
				-0.3941*sin(18*pi*t/T+1.2685)+0.2833*sin(20*pi*t/T+0.6702)+0.2272*sin(22*pi*t/T-1.4983)+0.2249*sin(24*pi*t/T+0.9924)
				+0.2589*sin(26*pi*t/T-1.5616)-0.1460*sin(28*pi*t/T-1.3106)+0.2141*sin(30*pi*t/T-1.1306)-0.1253*sin(32*pi*t/T+0.1552)
				+0.1321*sin(34*pi*t/T-1.5595)-0.1399*sin(36*pi*t/T+0.4223)-0.0324*sin(38*pi*t/T+0.7811)-0.1211*sin(40*pi*t/T+1.0729)
				)/1000/60
		elif self.FLUX_ID == 3:
			#t += 0.055;
			T = self.T_PER
			# shift = 0.0;//-3.796+0.0006082;
			# flux in ml/s:
			return 6.5+3.294*sin(2*pi*t/T-0.023974)+1.9262*sin(4*pi*t/T-1.1801)-1.4219*sin(6*pi*t/T+0.92701)
			-0.66627*sin(8*pi*t/T-0.24118)-0.33933*sin(10*pi*t/T-0.27471)-0.37914*sin(12*pi*t/T-1.0557)
			+0.22396*sin(14*pi*t/T+1.22)+0.1507*sin(16*pi*t/T+1.0984)+0.18735*sin(18*pi*t/T+0.067483)
			+0.038625*sin(20*pi*t/T+0.22262)+0.012643*sin(22*pi*t/T-0.10093)-0.0042453*sin(24*pi*t/T-1.1044)
			-0.012781*sin(26*pi*t/T-1.3739)+0.014805*sin(28*pi*t/T+1.2797)+0.012249*sin(30*pi*t/T+0.80827)
			+0.0076502*sin(32*pi*t/T+0.40757)+0.0030692*sin(34*pi*t/T+0.195)-0.0012271*sin(36*pi*t/T-1.1371)
			-0.0042581*sin(38*pi*t/T-0.92102)-0.0069785*sin(40*pi*t/T-1.2364)+0.0085652*sin(42*pi*t/T+1.4539)
			+0.0081881*sin(44*pi*t/T+0.89599)+0.0056549*sin(46*pi*t/T+0.17623)+0.0026358*sin(48*pi*t/T-1.3003)
			-0.0050868*sin(50*pi*t/T-0.011056)-0.0085829*sin(52*pi*t/T-0.86463)

	def data_flux(self, t):
		return self.FH.flux(t)

	def flux(self, t):
		if self.FLUX_TYPE == 'analytics':
			return self.analytic_flux(t)
		else:
			return self.data_flux(t)

	def compute_junction_bc_bernoulli(self, model):
		N = len(self.indices)
		JJ = np.empty(shape=(N,N))
		RESID = np.zeros(shape=N)
		alphas, betas = np.empty(shape=N), np.empty(shape=N)
		AREA = np.empty(shape=N)
		ID = [0]*N
		ORI = np.empty(shape=N)
		vessel_ids, bc_sides = [0]*N, ['a']*N
		for i in range(N):
			vessel_ids[i], bc_sides[i] = self.indices[i]
			alphas[i], betas[i] = self.compute_bc_coefs(model, i)
			vsl = model.vessels[vessel_ids[i]]
			ID[i] = 0 if bc_sides[i] == 'left' else vsl.N-1
			ORI[i] = 1 if bc_sides[i] == 'right' else -1
			AREA[i] = vsl.S[ID[i]]
		#import pdb; pdb.set_trace()
		# NONLINEAR LOOP, NEWTON METHOD
		rnorm0, rnorm = 0., 0.
		niters, niters_max = 0, 1000
		fail = False
		RHO = model.RHO
		while True:
			# COMPUTE RESIDUAL
			rnorm = 0.
			P_RHO = np.array([model.vessels[vessel_ids[i]].pressure(AREA[i]) for i in range(N)]) + RHO/2 * np.power(alphas*AREA+betas, 2)
			RESID[1:] = P_RHO[0] - P_RHO[1:]
			RESID[0] = (ORI*AREA*(alphas*AREA+betas)).sum()
			rnorm = sqrt(RESID.dot(RESID))
			if rnorm0 == 0.:
				rnorm0 = rnorm
			# input(f'iter {niters} rnorm {rnorm} rnorm0 {rnorm0} rnorm/rnorm0 {rnorm/rnorm0} RESID {RESID}')
			# FINISH SUCCESSFULLY, IF THE RESIDUAL IS SMALL
			if rnorm < 1.0e-9 or rnorm/rnorm0 < 1.0e-4:
				fail = False
				break
			# BREAK UNSUCCESSFULLY IF AMOUNT OF ITERATIONS IS BIG OR RESIDUAL IS BIG
			if niters > niters_max or rnorm/rnorm0 > 1.0e10 or rnorm > 1.0e12:
				fail = True
				break
			# COMPOSE JACOBIAN
			JJ[0] = ORI*(2*alphas*AREA+betas)
			# FILL ROW i
			for i in range(1,N):
				JJ[i][0] =  model.vessels[vessel_ids[0]].pressure_deriv(AREA[0]) + RHO * alphas[0] * (alphas[0]*AREA[0]+betas[0])
				JJ[i][i] = -(model.vessels[vessel_ids[i]].pressure_deriv(AREA[i]) + RHO * alphas[i] * (alphas[i]*AREA[i]+betas[i]))
			if np.any(np.isnan(RESID)) or np.any(np.isinf(RESID)):
				input(f'T {model.T} bad values: inf {np.any(np.isinf(RESID))} nan {np.any(np.isnan(RESID))} area {AREA}')
			# SOLVE SYSTEM
			X = lu_solve( lu_factor(JJ), RESID)
			xnorm = sqrt(X.dot(X))
			# UPDATE AREA
			AREA -= X
			if np.any(AREA <= 0.0):
				import pdb; pdb.set_trace()
			assert np.all(AREA > 0.0), 'negative area'
			niters += 1
			if abs(xnorm) < 1.0e-15:
				fail = rnorm < 1.0e-9 or rnorm/rnorm0 < 1.0e-4
				break
		if not fail:
			Sp, Sm = 0.0, 0.0
			for i in range(N):
				vsl = model.vessels[vessel_ids[i]]
				vsl.S[ID[i]], vsl.U[ID[i]] = AREA[i], alphas[i]*AREA[i]+betas[i]
				if model.USE_C:
					EPS_U = ORI[i]*vsl.U[ID[i]]
					i0, i1 = (0, 1) if bc_sides[i] == 'left' else (vsl.N-2, vsl.N-1)
					if (EPS_U > 1.0e-12 and bc_sides[i] == 'right') or (EPS_U < -1.0e-12 and bc_sides[i] == 'left'):
						# flux is directed from the inner side of the vessel, solve transport equation in boundary node
						vsl.C[ID[i]] -= model.DT/model.DX * ORI[i]*vsl.U[ID[i]] * (vsl.C[i1] - vsl.C[i0])
						Sp += ORI[i]*vsl.C[ID[i]]*vsl.S[ID[i]]*vsl.U[ID[i]]
					elif (EPS_U > 1.0e-12 and bc_sides[i] == 'left') or (EPS_U < -1.0e-12 and bc_sides[i] == 'right'):
						# flux is directed from the outside, compute sum of incoming fluxes in junction
						Sm += ORI[i]*vsl.S[ID[i]]*vsl.U[ID[i]]
			if model.USE_C and np.isfinite(Sm) and abs(Sm) > 1.0e-30:
				# mass balance
				# see (2.60)-(2.63) in articles/yns_phd.pdf
				for i in range(N):
					vsl = model.vessels[vessel_ids[i]]
					EPS_U = ORI[i]*(alphas[i]*AREA[i]+betas[i])
					if (EPS_U > 1.0e-12 and bc_sides[i] == 'left') or (EPS_U < -1.0e-12 and bc_sides[i] == 'right'):
						vsl.C[ID[i]] = -Sp/Sm
		else:
			input(f'Fail to solve nonlinear system for junction BC; T {model.T} iters {niters}\
			 rnorm {rnorm} rnorm0 {rnorm0} rnorm/rnorm0 {rnorm/rnorm0} xnorm {xnorm}')

	def compute_stenosis_bc_pinn(self, model):
		N = len(self.indices)
		JJ = np.empty(shape=(N, N))
		RESID = np.zeros(shape=N)
		alphas, betas = np.empty(shape=N), np.empty(shape=N)
		AREA = np.empty(shape=N)
		ID = [0] * N
		ORI = np.empty(shape=N)
		vessel_ids, bc_sides = [0] * N, ['a'] * N

		for i in range(N):
			vessel_ids[i], bc_sides[i] = self.indices[i]
			alphas[i], betas[i] = self.compute_bc_coefs(model, i)
			vsl = model.vessels[vessel_ids[i]]
			ID[i] = 0 if bc_sides[i] == 'left' else vsl.N - 1
			ORI[i] = 1 if bc_sides[i] == 'right' else -1
			AREA[i] = vsl.S[ID[i]]

		assert N == 2, '2 vessels'

		pinn = get_pinn()
		RHO = model.RHO

		COEF1 = 7.5e-4
		COEF2 = 7.5e-6
		rho_phys = model.RHO / COEF1       # g/cm^3
		mu_mPa_s = model.MU / COEF2        # mPa*s
		mu_phys = mu_mPa_s * 0.01          # g/(cm*s)

		R_ref = float(self.R_ref)
		Lr = float(self.Lr)
		Ds = float(self.Ds)
		asym = float(self.asym if 'asym' in self else 0.0)

		if R_ref <= 0.0:
			raise ValueError("R_ref must be positive")

		def dp_from_u(u_signed):
			u_eps = 1.0e-4
			u_abs_smooth = np.sqrt(u_signed * u_signed + u_eps * u_eps)
			Re = 2.0 * rho_phys * u_abs_smooth * R_ref / max(mu_phys, 1.0e-30)

			return float(
				pinn.dp_mmhg_from_Re(
					Re=Re,
					Lr=Lr,
					Ds=Ds,
					rho_phys=rho_phys,
					mu_phys=mu_phys,
					R_ref=R_ref,
					asym=asym,
					signed_by_Q=u_signed,
				)
			)
		def resid_for(area):
			u_end = alphas * area + betas
			q_through = ORI[0] * area[0] * u_end[0]   

			u_pinn = ORI[0] * u_end[0]          

			u_eps = 1.0e-4
			u_abs_smooth = np.sqrt(u_pinn * u_pinn + u_eps * u_eps)

			Re = 2.0 * rho_phys * u_abs_smooth * R_ref / max(mu_phys, 1.0e-30)

			dp_extra_mmhg = dp_from_u(u_pinn)

			P_RHO = np.array(
				[model.vessels[vessel_ids[i]].pressure(area[i]) for i in range(N)]
			) + RHO / 2.0 * np.power(alphas * area + betas, 2)

			resid = np.zeros(shape=N)
			resid[0] = (ORI * area * (alphas * area + betas)).sum()
			resid[1] = P_RHO[0] - P_RHO[1] - dp_extra_mmhg
			return resid.astype(float), float(dp_extra_mmhg), float(q_through), float(Re)

		rnorm0, rnorm = 0.0, 0.0
		niters, niters_max = 0, 1000
		fail = False
		xnorm = 0.0

		while True:
			RESID[:], dp_extra_mmhg, q_through, Re_now = resid_for(AREA)
			rnorm = sqrt(RESID.dot(RESID))

			if not np.isfinite(rnorm):
				fail = True
				break

			if rnorm0 == 0.0:
				rnorm0 = max(rnorm, 1.0e-30)

			ratio = rnorm / rnorm0

			if rnorm < 1.0e-4 or ratio < 1.0e-4:
				fail = False
				break

			if niters >= niters_max or ratio > 1.0e10 or rnorm > 1.0e12:
				fail = True
				break

			JJ[0] = ORI * (2.0 * alphas * AREA + betas)


			u0 = ORI[0] * (alphas[0] * AREA[0] + betas[0])
			du0_dA0 = ORI[0] * alphas[0]

			eps_u = max(1.0e-6, 1.0e-3 * max(abs(u0), 1.0e-3))

			dp_plus = dp_from_u(u0 + eps_u)
			dp_minus = dp_from_u(u0 - eps_u)

			ddp_dU = (dp_plus - dp_minus) / (2.0 * eps_u)

			for i in range(1, N):
				JJ[i][0] = (
					model.vessels[vessel_ids[0]].pressure_deriv(AREA[0])
					+ RHO * alphas[0] * (alphas[0] * AREA[0] + betas[0])
					- ddp_dU * du0_dA0
				)
				JJ[i][i] = -(
					model.vessels[vessel_ids[i]].pressure_deriv(AREA[i])
					+ RHO * alphas[i] * (alphas[i] * AREA[i] + betas[i])
				)

			if (
				np.any(np.isnan(RESID)) or np.any(np.isinf(RESID)) or
				np.any(np.isnan(JJ)) or np.any(np.isinf(JJ))
			):
				fail = True
				break

			try:
				X = lu_solve(lu_factor(JJ), RESID)
			except Exception:
				fail = True
				break

			if np.any(np.isnan(X)) or np.any(np.isinf(X)):
				fail = True
				break

			xnorm = sqrt(X.dot(X))

			AREA_prev = AREA.copy()
			step_scale = 1.0
			accepted = False

			while step_scale >= 1.0e-8:
				AREA_try = AREA_prev - step_scale * X

				if np.any(AREA_try <= 1.0e-12):
					step_scale *= 0.5
					continue

				RESID_try, _, _, _ = resid_for(AREA_try)
				rnorm_try = sqrt(RESID_try.dot(RESID_try))

				if np.isfinite(rnorm_try) and (rnorm_try < rnorm):
					AREA = AREA_try
					accepted = True
					break

				step_scale *= 0.5

			if not accepted:
				fail = True
				break
			
			niters += 1

			if abs(xnorm) < 1.0e-15:
				fail = not (rnorm < 1.0e-5 or ratio < 1.0e-4)
				break

		if fail:
			if "_last_pinn_fallback_print_t" not in model.__dict__:
				model._last_pinn_fallback_print_t = -1.0

			t_mark = round(model.T, 2)
			if abs(model.T - t_mark) < 0.5 * model.DT and t_mark != model._last_pinn_fallback_print_t:
				ratio = rnorm / max(rnorm0, 1.0e-30)
				print(
					f"FALLBACK to junction at T={model.T:.5f}, "
					f"iters={niters}, rnorm={rnorm:.3e}, ratio={ratio:.3e}, xnorm={xnorm:.3e}"
				)

			model._last_pinn_fallback_print_t = t_mark
			self.compute_junction_bc_bernoulli(model)
			return

		RESID_final, dp_extra_mmhg, q_through, Re_now = resid_for(AREA)


		if "_last_pinn_print_t" not in model.__dict__:
			model._last_pinn_print_t = -1.0

		A_ref = pi * R_ref ** 2
		V_ref = abs(q_through) / max(A_ref, 1.0e-30)
		P_dbg = np.array([model.vessels[vessel_ids[i]].pressure(AREA[i]) for i in range(N)])
		u_pinn_dbg = ORI[0] * (alphas[0] * AREA[0] + betas[0])


		Re_from_Vref = 2.0 * rho_phys * abs(V_ref) * R_ref / max(mu_phys, 1.0e-30)
		Re_from_u = 2.0 * rho_phys * abs(u_pinn_dbg) * R_ref / max(mu_phys, 1.0e-30)
		
		t_mark = round(model.T, 2)
		if abs(model.T - t_mark) < 0.5 * model.DT and t_mark != model._last_pinn_print_t:
			# print(
			# 	f"T={model.T:.2f}  "
			# 	f"Q={q_through:.4f}  "
			# 	f"Vref=Q/Aref={V_ref:.4f}  "
			# 	f"u_pinn={u_pinn_dbg:.4f}  "
			# 	f"Re_u={Re_from_u:.2f}  "
			# 	f"Re_Q={Re_from_Vref:.2f}  "
			# 	f"resid_Q = {RESID_final[0]}  "
			# 	f"resid_P = {RESID_final[1]}  "
			# 	f"P0={P_dbg[0]:.3f}  P1={P_dbg[1]:.3f}  "
			# 	f"dP={P_dbg[0] - P_dbg[1]:.3f}  "
			# 	f"dp_PINN={dp_extra_mmhg:.3f}  "
			# 	f"rnorm={rnorm:.3e}  "
			# 	f"ratio={ratio:.3e}  "
			# 	# f"iters={niters}"
			# )
			model._last_pinn_print_t = t_mark

		Sp, Sm = 0.0, 0.0
		for i in range(N):
			vsl = model.vessels[vessel_ids[i]]
			vsl.S[ID[i]], vsl.U[ID[i]] = AREA[i], alphas[i] * AREA[i] + betas[i]

			if model.USE_C:
				EPS_U = ORI[i] * vsl.U[ID[i]]
				i0, i1 = (0, 1) if bc_sides[i] == 'left' else (vsl.N - 2, vsl.N - 1)

				if (EPS_U > 1.0e-12 and bc_sides[i] == 'right') or (EPS_U < -1.0e-12 and bc_sides[i] == 'left'):
					vsl.C[ID[i]] -= model.DT / model.DX * ORI[i] * vsl.U[ID[i]] * (vsl.C[i1] - vsl.C[i0])
					Sp += ORI[i] * vsl.C[ID[i]] * vsl.S[ID[i]] * vsl.U[ID[i]]
				elif (EPS_U > 1.0e-12 and bc_sides[i] == 'left') or (EPS_U < -1.0e-12 and bc_sides[i] == 'right'):
					Sm += ORI[i] * vsl.S[ID[i]] * vsl.U[ID[i]]

		if model.USE_C and np.isfinite(Sm) and abs(Sm) > 1.0e-30:
			for i in range(N):
				vsl = model.vessels[vessel_ids[i]]
				EPS_U = ORI[i] * (alphas[i] * AREA[i] + betas[i])
				if (EPS_U > 1.0e-12 and bc_sides[i] == 'left') or (EPS_U < -1.0e-12 and bc_sides[i] == 'right'):
					vsl.C[ID[i]] = -Sp / Sm

	def compute_bc(self, model):
		# JUNCTION
		if self.BC_TYPE == 'junction': # BERNOULLI
			self.compute_junction_bc_bernoulli(model)
		elif self.BC_TYPE == 'stenosis_pinn':
			self.compute_stenosis_bc_pinn(model)
		else:
			#import pdb; pdb.set_trace()
			v_id, bc_side = self.indices[0]
			vessel = model.vessels[v_id]
			alpha, beta = self.compute_bc_coefs(model, 0)
			i = 0 if bc_side == 'left' else vessel.N-1
			if self.BC_TYPE == 'inlet':
				q = self.flux(model.T + model.DT)
				vessel.S[i] = ( -beta + sqrt(beta*beta + 4*alpha*q) ) / (2 * alpha)
				# assert vessel.S[i] > 0.0, 'negative area'
				# assert abs(vessel.S[i]*vessel.U[i] - q) < 1.0e-9, 'fail to setup inflow BC'
				if model.USE_C:
					vessel.C[i] = self.C_INLET
			elif self.BC_TYPE == 'outlet':
				if self.BC_OUT == 'outflow_pressure':
					vessel.S[i] = vessel.get_area(self.P_OUT)
				elif self.BC_OUT == 'windkessel':
					# Newton method
					s0, s1, u0 = [vessel.S[i]][0], [vessel.S[i]][0], [vessel.U[i]][0]
					resid0 = self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta)
					resid, ds = [resid0][0], 1.0
					iters, fail = 0, False
					while True:
						if not (abs(ds) > 1.0e-12 and abs(resid) > 1.0e-15 and abs(resid/resid0) > 1.0e-12):
							fail = False
							# assert s1 > 0.0, 'negative area'
							break
						ds = -self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta) \
						/ self.wk_newton_dfds(vessel, model.DT, s1, s0, u0, alpha, beta)
						s1 += ds
						resid = self.wk_newton_f(vessel, model.DT, s1, s0, u0, alpha, beta)
						iters += 1
						if s1 < 0.0:
							fail = True
							break
					vessel.S[i] = s1
				#set BC on outlet to update boundary concentration which can be used and exported
				if model.USE_C:
					vessel.U[i] = alpha*vessel.S[i] + beta
					EPS_U = vessel.U[i]
					i0, i1 = (0, 1) if bc_side == 'left' else (vessel.N-2, vessel.N-1)
					if (EPS_U > 1.0e-12 and bc_side == 'right') or (EPS_U < -1.0e-12 and bc_side == 'left'):
						# flux is directed from the inner side of the vessel, solve transport equation in boundary node
						vessel.C[i] -= model.DT/model.DX * vessel.U[i] * (vessel.C[i1] - vessel.C[i0])
					elif (EPS_U > 1.0e-12 and bc_side == 'left') or (EPS_U < -1.0e-12 and bc_side == 'right'):
						# flux is directed from the outflow! assume zero Neumann BC
						pass
			vessel.U[i] = alpha*vessel.S[i] + beta


class FlowModel():
	def __init__(self, filename):
		with open(filename, 'r') as f:
			data = json.load(f)
		self.T = 0.0
		self.T_FINAL = data['total_time']
		self.RHO = data['density']
		self.MU = data['viscosity']
		self.DX = data['space_step']
		self.DT = data['time_step']
		self.bc_order = data['bc_order']
		self.SCHEME_GAMMA = data['scheme_gamma']
		self.ALASTRUEY_GAMMA = data['alastruey_gamma']
		self.CFL = data['CFL']
		self.USE_C = data['compute_concentration'] if 'compute_concentration' in data else False
		self.vessels = []
		self.savefig = data['savefig'] if 'savefig' in data else 'out0.png'
		for i in data['vessel'].keys():
			self.vessels.append(Vessel(data, i))
			self.vessels[-1].post_setup(self)
		self.pool = ThreadPool(len(self.vessels))
		print(f'CFL {self.CFL} time step: {self.DT}')
		self.bcs = []
		for i in data['bc'].keys():
			self.bcs.append(BC(data['bc'][i]))
			self.bcs[-1].post_setup(self)
		# process junction BC and provide correct orientations
		mark = np.zeros(shape=len(self.bcs))
		for i in range(len(self.bcs)):
			if self.bcs[i].BC_TYPE in ('junction', 'stenosis_pinn'):
				mark[i] = 1
		while True:
			all_processed = True
			for i in range(len(self.bcs)):
				if mark[i]:
					all_processed = False
					# one of two vessel sides on which BC is not set yet
					k = [0,'']
					for j in range(len(self.bcs[i].indices)):
						k1 = self.bcs[i].indices[j] # k1 = (v_id,bc_side)
						# bc_side indicates left or right side
						# of the vessel with index bcs[i].V_ID in FlowModel global array of vessels
						if self.vessels[k1[0]].bc_types[0] != '':
							k1[1] = 'right'
							k = k1
						elif self.vessels[k1[0]].bc_types[1] != '':
							k1[1] = 'left'
							k = k1
					if k[1] != '':
						for j in range(len(self.bcs[i].indices)):
							k2 = self.bcs[i].indices[j]
							# set bc_side where it is not set:
							if k2[1] != '' and self.vessels[k2[0]].bc_types[1 if k2[1] == 'right' else 0] != '':
								assert k2[1] == k[1], 'Bug or error in vessel connectivity setup'
							else:
								if k2[0] != k[0]:
									k2[1] = 'left' if k[1] == 'right' else 'right'
								#k2[1] = (-1 if k2[0] != k[0] else 1) * k[1]
								# vessel is processed at this end
								self.vessels[k2[0]].bc_types[0 if k2[1] == 'left' else 1] = 'junction'
						# junction BC is processed
						mark[i] = 0
			if all_processed:
				break
		# setup outlet BC on remaining unset vessel ends
		for i in range(len(self.bcs)):
			if self.bcs[i].BC_TYPE == 'outlet':
				k1 = self.bcs[i].indices[0] # (v_id, bc_side)
				if self.vessels[k1[0]].bc_types[0] == '':
					self.vessels[k1[0]].bc_types[0] = 'outlet'
					k1[1] = 'left'
				elif self.vessels[k1[0]].bc_types[1] == '':
					self.vessels[k1[0]].bc_types[1] = 'outlet'
					k1[1] =  'right'
				else:
					assert False, 'Fail to setup outlet BC, possibly error in arterial network topology?.'
			if self.bcs[i].BC_TYPE != 'junction':
				k2 = self.bcs[i].indices[0]
				print(f'info: i {i} type {self.bcs[i].BC_TYPE} side {k2[1]} vessel {k2[0]} '
					f'bc_types {self.vessels[k2[0]].bc_types[0]} {self.vessels[k2[0]].bc_types[1]}')
			else:
				print(f'info: i {i} type {self.bcs[i].BC_TYPE} sides {[self.bcs[i].indices[j][1] for j in range(len(self.bcs[i].indices))]} '
					f'vessels {[self.bcs[i].indices[j][0] for j in range(len(self.bcs[i].indices))]} info {self.bcs[i].indices}')


	def step_inner_points(self):
		#for v in self.vessels:
		#	v.step_inner_points()
		self.pool.map(Vessel.step_inner_points, self.vessels)

	def step(self):
		self.step_inner_points()
		for i in range(len(self.bcs)):
			self.bcs[i].compute_bc(self)
		self.T += self.DT

	def close(self):
		pool = getattr(self, "pool", None)
		if pool is not None:
			try:
				pool.close()
				pool.join()
			except Exception:
				pass
			self.pool = None

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass

	def run(self):
		steps = 0
		nsteps = int(self.T_FINAL / self.DT)
		nv = len(self.vessels)
		cnt, T_cnt = 0, 1.0e-3
		T_last = self.T_FINAL - self.bcs[0].T_PER
		nsave = int(self.T_FINAL / T_cnt)+1
		Pmid = np.zeros(shape=(2*nv,nsave))
		Pavg = np.zeros(shape=nv)
		imid = np.array( [ int(self.vessels[i].N/2) for i in range(nv) ])
		# imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
		# imid[2] = (int) (0.5 / self.vessels[2].L * self.vessels[2].N)
		if nv >= 1:
			imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
		if nv >= 2:
			imid[-1] = (int) (0.5 / self.vessels[-1].L * self.vessels[-1].N)
		outfile = open('p.csv', 'w')
		while self.T < self.T_FINAL-1.0e-12:
			self.step()
			steps += 1
			if abs(self.T - (cnt+1)*T_cnt) < 0.1*self.DT:
				print(f'{self.T:.04g}/{self.T_FINAL}', end='\r')
				for i in range(nv):
					Pmid[i][cnt] = self.vessels[i].pressure(self.vessels[i].S[imid[i]])
					Pmid[nv+i][cnt] = self.vessels[i].U[imid[i]] * self.vessels[i].S[imid[i]]
				if self.T >= T_last:
					# outfile.write(f'{self.T-T_last:.3f};{Pmid[0][cnt]:.3f};{Pmid[1][cnt]:.3f};{Pmid[2][cnt]:.3f}\n')
					vals = ';'.join([f'{Pmid[i][cnt]:.3f}' for i in range(nv)])
					outfile.write(f'{self.T-T_last:.3f};{vals}\n')
				cnt += 1
				if self.USE_C:
					fig,axs = plt.subplots(2,3)
					nshow = min(nv, 6)
					for i in range(nshow):
						C1 = self.vessels[i].U
						#C1 = self.vessels[i].pressure(self.vessels[i].S) if i < 3 else self.vessels[0].U
						axs[i//3,i%3].plot(np.linspace(0,self.vessels[i if i < 6 else 0].L,C1.size), C1, '-o')
						axs[i//3,i%3].grid()
					plt.draw()
					#plt.ylim(-0.01+np.min(C1),0.01+np.max(C1))
					plt.savefig(f'tmp/C0/{cnt:04d}.png')
					plt.clf()
					plt.close()
			if self.T >= T_last:
				for i in range(nv):
					Pavg[i] += self.vessels[i].pressure(self.vessels[i].S[imid[i]]) * self.DT
		for i in range(nv):
			Pavg[i] /= (self.T-T_last)
		outfile.close()
		# x = np.linspace(0, self.T, nsave)
		# print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[2]} FFR = Pavg2/Pavg0 = {Pavg[2]/Pavg[0]}')
		x = np.arange(nsave) * T_cnt
		if nv >= 4:
			print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[-1]} FFR = Pout/Pin = {Pavg[-1]/Pavg[0]}')
		elif nv >= 3:
			print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[2]} FFR = Pavg2/Pavg0 = {Pavg[2]/Pavg[0]}')
		elif nv >= 2:
			print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[1]} FFR = Pout/Pin = {Pavg[1]/Pavg[0]}')
		else:
			print(f'Pavg: {Pavg}')
		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')
		# plt.plot(x, Pmid[0], label='Pmid_vessel_0', color='r')
		# plt.plot(x, Pmid[1], label='Pmid_vessel_1', color='g')
		# plt.plot(x, Pmid[2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[3+0], label='Pmid_vessel_0', color='r')
		#plt.plot(x, Pmid[3+1], label='Pmid_vessel_1', color='g')
		#plt.plot(x, Pmid[3+2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[0]-Pmid[2], label='dPmid_vessel0-vessel2', color='k')
		#plt.xlim(T_last, self.T)
		#plt.ylim(75.0, 165.0)
		for i in range(nv):
			plt.plot(x, Pmid[i], label=f'Pmid_vessel_{i}')
		plt.grid(visible=True)
		plt.legend()
		plt.savefig(self.savefig)
