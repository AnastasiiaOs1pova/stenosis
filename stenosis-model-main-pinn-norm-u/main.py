from model import FlowModel
import sys
import numpy as np
import matplotlib.pyplot as plt

class MyModel(FlowModel):
	def run(self):
		steps = 0
		nsteps = int(self.T_FINAL / self.DT)
		cnt, T_cnt = 0, 1.0e-3
		T_last = self.T_FINAL - self.bcs[0].T_PER
		nsave = int(self.T_FINAL / T_cnt)+2

		nv = len(self.vessels)
		if nv == 0:
			raise RuntimeError('No vessels found in the model.')

		# Pmid = np.zeros(shape=(6,nsave))
		# Pavg = np.zeros(shape=3)
		Pmid = np.zeros(shape=(2*nv,nsave))
		Pavg = np.zeros(shape=nv)
		times = np.zeros(shape=nsave)

		# imid = np.array( [ int(self.vessels[i].N/2) for i in range(3) ])
		imid = np.array( [ int(self.vessels[i].N/2) for i in range(nv) ])

		# imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
		# imid[2] = (int) (0.5 / self.vessels[2].L * self.vessels[2].N)
		if nv >= 1 and self.vessels[0].L > 0:
			imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
			imid[0] = max(0, min(imid[0], self.vessels[0].N-1))
		if nv >= 2 and self.vessels[-1].L > 0:
			imid[-1] = (int) (0.5 / self.vessels[-1].L * self.vessels[-1].N)
			imid[-1] = max(0, min(imid[-1], self.vessels[-1].N-1))

		# outfile = open('p.csv', 'w')
		with open('p.csv', 'w', encoding='utf-8') as outfile:
			header = ['time'] + [f'P_vessel_{i}' for i in range(nv)]
			outfile.write(';'.join(header) + '\n')

			while self.T < self.T_FINAL-1.0e-12:
				self.step()
				steps += 1
				if abs(self.T - (cnt+1)*T_cnt) < 0.1*self.DT:
					print(f'{self.T:.04g}/{self.T_FINAL}', end='\r')
					times[cnt] = self.T

					# for i in range(3):
					# 	Pmid[i][cnt] = self.vessels[i].pressure(self.vessels[i].S[imid[i]])
					# 	Pmid[3+i][cnt] = self.vessels[i].U[imid[i]] * self.vessels[i].S[imid[i]]
					for i in range(nv):
						Pmid[i][cnt] = self.vessels[i].pressure(self.vessels[i].S[imid[i]])
						Pmid[nv+i][cnt] = self.vessels[i].U[imid[i]] * self.vessels[i].S[imid[i]]

					if self.T >= T_last:
						# outfile.write(f'{self.T-T_last:.3f};{Pmid[0][cnt]:.3f};{Pmid[1][cnt]:.3f};{Pmid[2][cnt]:.3f}\n')
						vals = ';'.join([f'{Pmid[i][cnt]:.3f}' for i in range(nv)])
						outfile.write(f'{self.T-T_last:.3f};{vals}\n')
					cnt += 1

				if self.T >= T_last:
					# for i in range(3):
					# 	Pavg[i] += self.vessels[i].pressure(self.vessels[i].S[imid[i]]) * self.DT
					for i in range(nv):
						Pavg[i] += self.vessels[i].pressure(self.vessels[i].S[imid[i]]) * self.DT

		# for i in range(3):
		for i in range(nv):
			Pavg[i] /= (self.T-T_last)

		self.close()

		# x = np.linspace(0, self.T, nsave)
		times = times[:cnt]
		Pmid = Pmid[:, :cnt]

		# print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[2]} FFR = Pavg2/Pavg0 = {Pavg[2]/Pavg[0]}')
		p_in = Pavg[0]
		p_out = Pavg[-1]
		dp = p_in - p_out
		ffr = p_out / p_in if abs(p_in) > 1.0e-12 else np.nan
		print()
		print(f'Pavg: {Pavg} dP {dp} FFR = Pout/Pin = {ffr}')

		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')
		# plt.plot(x, Pmid[0], label='Pmid_vessel_0', color='r')
		# plt.plot(x, Pmid[1], label='Pmid_vessel_1', color='g')
		# plt.plot(x, Pmid[2], label='Pmid_vessel_2', color='b')
		for i in range(nv):
			plt.plot(times, Pmid[i], label=f'Pmid_vessel_{i}')
		#plt.plot(x, Pmid[3+0], label='Pmid_vessel_0', color='r')
		#plt.plot(x, Pmid[3+1], label='Pmid_vessel_1', color='g')
		#plt.plot(x, Pmid[3+2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[0]-Pmid[2], label='dPmid_vessel0-vessel2', color='k')
		#plt.xlim(T_last, self.T)
		#plt.ylim(75.0, 165.0)
		plt.grid(visible=True)
		plt.legend()
		plt.savefig('out0.png', dpi=150, bbox_inches='tight')
		plt.close()

		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f'Usage: python3 {sys.argv[0]} run/test.json')
		sys.exit()
	model = MyModel(sys.argv[1])
	model.run()