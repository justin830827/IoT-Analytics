'''
This script is for CSC591 IoT Analytics

Title: Simulation Task 2.1
Author: Wen-Han (Justin) Hu (whu24)

'''

def simulation(mean_at_RT,mean_at_nonRT,mean_st_RT,mean_st_nonRT):
	MC, RTCL, nonRTCL, n_RT, n_nonRT, SCL, status, preempt = [0], [3], [5], [0], [0], [4], [2], [0]
	f = open("task2.1_output.txt", "a")
	f.write("\t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t {}\t\t|\n".format(MC[-1], RTCL[-1], nonRTCL[-1], n_RT[-1], n_nonRT[-1], SCL[-1], status[-1], preempt[-1]))
	f.close()
	event = [RTCL[-1],nonRTCL[-1],SCL[-1]] # 0: SCL, 1: RT, 2: nonRT
	# simulate MC until MC > 50
	while MC[-1] < 200:
		index = event.index(min(event))
		if index == 2 and n_RT[-1] == 0 and n_nonRT[-1] == 0 and preempt[-1] == 0:
			event.pop(index)
			index = event.index(min(event))
		# if RT arrive
		if index == 0:
			MC.append(RTCL[-1])
			RTCL.append(RTCL[-1]+mean_at_RT)
			nonRTCL.append(nonRTCL[-1])
			n_RT.append(n_RT[-1] + (1 if SCL[-1] > MC[-1] and status[-1] == 1 else 0))
			n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > RTCL[-2] else 0))
			SCL.append(MC[-1]+mean_st_RT)
			status.append(1)
			preempt.append(SCL[-2]-RTCL[-2] if (SCL[-2]-RTCL[-2]) > 0 else 0)
			event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
		# if nonRT arrive
		elif index == 1:
			MC.append(nonRTCL[-1])
			RTCL.append(RTCL[-1])
			nonRTCL.append(nonRTCL[-1]+mean_at_nonRT)
			n_RT.append(n_RT[-1])
			n_nonRT.append(n_nonRT[-1] + 1)
			SCL.append(SCL[-1]) 
			status.append(status[-1])
			preempt.append(preempt[-1])
			event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
		else:
			# if scl is min, server is idle, check n_RT first and then check nonRT
			if n_RT[-1] > 0:
				MC.append(RTCL[-1])
				RTCL.append(RTCL[-1])
				nonRTCL.append(nonRTCL[-1])
				n_RT.append(n_RT[-1] -1)
				n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > RTCL[-2] else 0))
				SCL.append(RTCL[-2]+mean_st_RT)
				status.append(1)
				preempt.append(SCL[-2]-RTCL[-2] if (SCL[-2]-RTCL[-2]) > 0 else 0)
				event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
			else: 
				# check preempt first, if there is no preempt, do non_RT
				if preempt[-1] > 0:
					MC.append(SCL[-1])
					RTCL.append(RTCL[-1])
					nonRTCL.append(nonRTCL[-1])
					n_RT.append(0)
					n_nonRT.append(n_nonRT[-1] - 1)
					SCL.append(SCL[-1]+preempt[-1]) 
					status.append(2)
					preempt.append(0)
					event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
				else:
					MC.append(SCL[-1])
					RTCL.append(RTCL[-1])
					nonRTCL.append(nonRTCL[-1])
					n_RT.append(0)
					n_nonRT.append(n_nonRT[-1] - 1)
					SCL.append(SCL[-1] + mean_st_nonRT) 
					status.append(2)
					preempt.append(0)
					event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
		f = open("task2.1_output.txt", "a")
		f.write("\t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t{}\t| \t {}\t\t|\n"
				.format(MC[-1], RTCL[-1], nonRTCL[-1], n_RT[-1], n_nonRT[-1], SCL[-1], status[-1], preempt[-1]))
		f.close()

if __name__ == "__main__":
	mean_at_RT = input("Please Enter mean inter-arrival time of RT messages:")
	mean_at_nonRT = input("Please Enter mean inter-arrival time of nonRT messages:")
	mean_st_RT = input("Please Enter mean service time of RT messages:")
	mean_st_nonRT = input("Please Enter mean service time of nonRT messages:")
	f = open("task2.1_output.txt", "w")
	f.write("\tMC\t| RTCL  |nonRTCL| n_RT  |n_nonRT|  SCL  | Status| Preempted\t|\n")
	f.close()
	simulation(int(mean_at_RT),int(mean_at_nonRT),int(mean_st_RT),int(mean_st_nonRT))




