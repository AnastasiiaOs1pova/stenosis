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
		nsave = int(self.T_FINAL / T_cnt)+1

		Pmid = np.zeros(shape=(6,nsave))
		Pavg = np.zeros(shape=3)
		imid = np.array( [ int(self.vessels[i].N/2) for i in range(3) ])
		imid[0] = (int) ( (1.0 - 0.5 / self.vessels[0].L) * self.vessels[0].N)
		imid[2] = (int) (0.5 / self.vessels[2].L * self.vessels[2].N)
		outfile = open('p.csv', 'w')

		while self.T < self.T_FINAL-1.0e-12:
			self.step()
			steps += 1
			if abs(self.T - (cnt+1)*T_cnt) < 0.1*self.DT:
				print(f'{self.T:.04g}/{self.T_FINAL}', end='\r')
				for i in range(3):
					Pmid[i][cnt] = self.vessels[i].pressure(self.vessels[i].S[imid[i]])
					Pmid[3+i][cnt] = self.vessels[i].U[imid[i]] * self.vessels[i].S[imid[i]]
				if self.T >= T_last:
					outfile.write(f'{self.T-T_last:.3f};{Pmid[0][cnt]:.3f};{Pmid[1][cnt]:.3f};{Pmid[2][cnt]:.3f}\n')
				cnt += 1
			if self.T >= T_last:
				for i in range(3):
					Pavg[i] += self.vessels[i].pressure(self.vessels[i].S[imid[i]]) * self.DT
		for i in range(3):
			Pavg[i] /= (self.T-T_last)
		outfile.close()
		x = np.linspace(0, self.T, nsave)
		print(f'Pavg: {Pavg} dP {Pavg[0]-Pavg[2]} FFR = Pavg2/Pavg0 = {Pavg[2]/Pavg[0]}')
		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')
		plt.plot(x, Pmid[0], label='Pmid_vessel_0', color='r')
		plt.plot(x, Pmid[1], label='Pmid_vessel_1', color='g')
		plt.plot(x, Pmid[2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[3+0], label='Pmid_vessel_0', color='r')
		#plt.plot(x, Pmid[3+1], label='Pmid_vessel_1', color='g')
		#plt.plot(x, Pmid[3+2], label='Pmid_vessel_2', color='b')
		#plt.plot(x, Pmid[0]-Pmid[2], label='dPmid_vessel0-vessel2', color='k')
		#plt.xlim(T_last, self.T)
		#plt.ylim(75.0, 165.0)
		plt.grid(visible=True)
		plt.savefig('out0.png')

		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f'Usage: python3 {sys.argv[0]} run/test.json')
		exit()
	model = MyModel(sys.argv[1])
	model.run()
