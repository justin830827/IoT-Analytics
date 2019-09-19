'''
This script is for CSC591 IoT Analytics

Title: Simulation Task 2.2
Author: Wen-Han (Justin) Hu (whu24)

'''
import random
import math
import texttable as tt

random.seed(100)

def simulation(mean_at_RT,mean_at_nonRT,mean_st_RT,mean_st_nonRT):
	MC, RTCL, nonRTCL, n_RT, n_nonRT, SCL, status, preempt = [0.0], [3.0], [5.0], [0], [0], [4.0], [2], [0.0]
	event = [RTCL[-1],nonRTCL[-1],SCL[-1]] # 0: SCL, 1: RT, 2: nonRT
	# simulate MC until MC > 50
	while MC[-1] < 200:
		r=random.uniform(0,1)
		rand_at_RT = -1*mean_at_RT*math.log(r)
		rand_at_nonRT = -1*mean_at_nonRT*math.log(r)
		rand_st_RT = -1*mean_st_RT*math.log(r)
		rand_st_nonRT = -1*mean_st_nonRT*math.log(r)
		index = event.index(min(event))
		if index == 2 and n_RT[-1] == 0 and n_nonRT[-1] == 0 and preempt[-1] == 0:
			event.pop(index)
			index = event.index(min(event))
		# if RT arrive
		if index == 0:
			MC.append(RTCL[-1])
			RTCL.append(RTCL[-1]+rand_at_RT)
			nonRTCL.append(nonRTCL[-1])
			n_RT.append(n_RT[-1] + (1 if SCL[-1] > MC[-1] and status[-1] == 1 else 0))
			n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > RTCL[-2] else 0))
			SCL.append(MC[-1]+rand_st_RT)
			status.append(1)
			preempt.append(SCL[-2]-RTCL[-2] if (SCL[-2]-RTCL[-2]) > 0 else 0)
			event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
		# if nonRT arrive
		elif index == 1:
			# if rt still execute or nonrt execute
			if nonRTCL[-1] < SCL[-1]:
				MC.append(nonRTCL[-1])
				RTCL.append(RTCL[-1])
				nonRTCL.append(nonRTCL[-1]+rand_at_nonRT)
				n_RT.append(n_RT[-1])
				n_nonRT.append(n_nonRT[-1] + 1)
				SCL.append(SCL[-1]) 
				status.append(status[-1])
				preempt.append(preempt[-1])
				event = [RTCL[-1],nonRTCL[-1],SCL[-1]]
			# rt done, nonrt is able to execute
			else:
				MC.append(nonRTCL[-1])
				RTCL.append(RTCL[-1])
				nonRTCL.append(nonRTCL[-1]+rand_at_nonRT)
				n_RT.append(n_RT[-1])
				n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1]>MC[-1] else 0))
				SCL.append(MC[-1]+rand_st_nonRT) 
				status.append(2)
				preempt.append(preempt[-1])
				event = [RTCL[-1],nonRTCL[-1],SCL[-1]] 

		# if server idle
		else:
			# check n_RT first and then check nonRT
			if n_RT[-1] > 0:
				MC.append(SCL[-1])
				RTCL.append(RTCL[-1])
				nonRTCL.append(nonRTCL[-1])
				n_RT.append(n_RT[-1] - 1)
				n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > RTCL[-2] else 0))
				SCL.append(RTCL[-2]+rand_st_RT)
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
					SCL.append(SCL[-1] + rand_st_nonRT) 
					status.append(2)
					preempt.append(0)
					event = [RTCL[-1],nonRTCL[-1],SCL[-1]]

	# print table
	tab = tt.Texttable()
	headings = ['MC','RTCL','nonRTCL','n_RT', 'n_nonRT','SCL',
				'Server Status','Pre-empted remaining service time']
	tab.header(headings)
	for row in zip(MC, RTCL, nonRTCL, n_RT, n_nonRT, SCL, status, preempt):
		tab.add_row(row)
	print (tab.draw(),file=open("task2.2_output.txt", "a"))


if __name__ == "__main__":
	mean_at_RT = input("Please Enter mean inter-arrival time of RT messages:")
	mean_at_nonRT = input("Please Enter mean inter-arrival time of nonRT messages:")
	mean_st_RT = input("Please Enter mean service time of RT messages:")
	mean_st_nonRT = input("Please Enter mean service time of nonRT messages:")
	simulation(int(mean_at_RT),int(mean_at_nonRT),int(mean_st_RT),int(mean_st_nonRT))




