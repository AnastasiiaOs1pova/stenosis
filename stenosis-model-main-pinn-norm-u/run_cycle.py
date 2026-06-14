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
		while self.T < self.T_FINAL-1.0e-12:
			self.step()
			steps += 1
			if abs(self.T - (cnt+1)*T_cnt) < 0.1*self.DT:
				print(f'{self.T:.04g}/{self.T_FINAL}', end='\r')
				cnt += 1
				if self.USE_C:
					fig,axs = plt.subplots(2,3)
					for i in range(1):
						dat = self.vessels[i].C
						axs[i//3,i%3].plot(np.linspace(0,self.vessels[i if i < 6 else 0].L,dat.size), dat, '-o')
						axs[i//3,i%3].grid()
					plt.draw()
					plt.savefig(f'tmp/C0/{cnt:04d}.png')
					plt.clf()
					plt.close()
		# for i in range(len(self.vessels)):
		#	 print(f'vessel {i} times_inner_points: {self.vessels[i].tinn}')


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f'Usage: python3 {sys.argv[0]} run/test.json')
		exit()
	model = MyModel(sys.argv[1])
	model.run()
