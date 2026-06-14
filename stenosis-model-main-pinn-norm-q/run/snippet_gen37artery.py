import math
import sys

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f'Usage: {sys.argv[0]} /path/to/benchmark37_artery_data/mesh.txt')
	fname = sys.argv[1]
	with open(fname) as f:
		s = []
		for ss in f:
			s.append(ss)
	strs = [x.split('\t') for x in s]
	pref = '\t\t'
	for i in range(len(strs)):
		length, d_in, d_out = float(strs[i][3])*100, float(strs[i][4])*200, float(strs[i][5])*200
		area_dia = math.pi*(d_in+d_out)**2 / 16
		h_wall, c_speed = float(strs[i][6])*100, float(strs[i][8])*100
		e_young = 9.0e3
		print(pref+f'"vessel_{str(i)}":')
		print(pref+f"{{")
		print(pref+f'\t"length": {length:0.5g},')
		print(pref+f'\t"diameter": {d_in:.4g},')
		print(pref+f'\t"diameter_out": {d_out:.4g},')
		print(pref+f'\t"area_diastolic": {area_dia:0.5g},')
		print(pref+f'\t"pressure_external": 0.0,')
		print(pref+f'\t"pressure_diastolic": 0.0,')
		print(pref+f'\t"perturbation_speed": {c_speed:4.1f},')
		print(pref+f'\t"wall_thickness": {h_wall:0.5g},')
		print(pref+f'\t"young_modulus": 9.0e3')
		print(pref+f'}},')

# NAMES=(AORTIC_ARCH_II THORATIC_AORTA_II L_SUBCLAVIAN_I R_ILIAC_FEMORAL_II L_ULNAR R_ANTERIOR_TIBIAL R_ULNAR SPLENIC)
# IND=(9 16 10 28 13 33 6 19)
# for d in pde ; do for i in $(seq 0 7); do j=${IND[i]}; f=$(ls $d/vessel_${j}_*.txt); echo "mv $f $d/${NAMES[${i}]}.txt"; done; done
# T_START=(14.075 14.09 14.097 14.162 14.138 14.2 14.136 14.134)