'''
This script is for CSC591 IoT Analytics

Title: Simulation Task 3
Author: Wen-Han (Justin) Hu (whu24)

'''
import random
import math
import csv

# Setup random seed
random.seed(100)

def count_percentile(data, percentile):
    size = len(data)
    return sorted(data)[int(math.ceil(size * percentile)) - 1]

def count_ci(batch_mean,std,m):
	t95 = 1.68
	value = t95 * std / math.sqrt(m)
	return (batch_mean - value, batch_mean + value)

def count_std(samples,batch_mean,batch_number):
	diff_sum = 0
	for s in samples:
		diff_sum += math.pow(s - batch_mean,2)
	return math.sqrt(diff_sum / (batch_number - 1))
		
def count_mean(samples):
	return sum(samples) / len(samples)

def simulation(mean_at_RT,mean_at_nonRT,mean_st_RT,mean_st_nonRT,m,b):
	# Initialize simulation params 
	MC, RTCL, nonRTCL, n_RT, n_nonRT, SCL, status, preempt = [0.0], [3.0], [5.0], [0], [0], [4.0], [2], [0.0]

	# Event queue, choose the min time as nexrt event, if RTCL == nonRTCL,  RTCL will be chosen first (higher priority)
	event = [ RTCL[-1], nonRTCL[-1], SCL[-1] ] # 0: SCL, 1: RT, 2: nonRT
	
	# Decalre batch means collectors
	R_RT_mean, R_nonRT_mean = [], []
	RT_95percent, nonRT_95percent = [], []
	at_RT, at_nonRT = [], []
	# Execute number of batch times
	for i in range(m):
		R_RT, R_nonRT = [], []
		# Execute one batch simulation:
		for j in range(b):		
			# Generate random numbers
			r = random.uniform(0,1)
			rand_at_RT = -1 * mean_at_RT * math.log(r)
			rand_at_nonRT = -1 * mean_at_nonRT * math.log(r)
			rand_st_RT = -1 * mean_st_RT * math.log(r)
			rand_st_nonRT = -1 * mean_st_nonRT * math.log(r)
			
			# Choose the next event
			index = event.index(min(event)) 
			if index == 2 and n_RT[-1] == 0 and n_nonRT[-1] == 0 and preempt[-1] == 0: # If server is idle, choose next min event
				event.pop(index)
				index = event.index(min(event))
			# if RT arrive
			if index == 0:
				MC.append(RTCL[-1])
				RTCL.append(RTCL[-1] + rand_at_RT)
				nonRTCL.append(nonRTCL[-1])
				n_RT.append(n_RT[-1] + (1 if SCL[-1] > MC[-1] and status[-1] == 1 else 0))
				n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > RTCL[-2] else 0))
				SCL.append(MC[-1] + rand_st_RT)
				status.append(1)
				preempt.append(SCL[-2] - RTCL[-2] if (SCL[-2] - RTCL[-2]) > 0 else 0)
				event = [ RTCL[-1], nonRTCL[-1], SCL[-1]]
				
				# Handle arrival RT and response RT
				at_RT.append(MC[-1])
				# If RT execute directly, response time = 0
				if n_RT[-1] == 0:
					at_RT.pop(0)
					R_RT.append(0)

			# if nonRT arrive
			elif index == 1:
				# if RT still execute or nonRT execute
				if nonRTCL[-1] < SCL[-1]:
					MC.append(nonRTCL[-1])
					RTCL.append(RTCL[-1])
					nonRTCL.append(nonRTCL[-1] + rand_at_nonRT)
					n_RT.append(n_RT[-1])
					n_nonRT.append(n_nonRT[-1] + 1)
					SCL.append(SCL[-1]) 
					status.append(status[-1])
					preempt.append(preempt[-1])
					event = [ RTCL[-1], nonRTCL[-1], SCL[-1]]

					# Handle arrival nonRT
					at_nonRT.append(MC[-1])

				# RT done, nonRT is able to execute
				else:
					MC.append(nonRTCL[-1])
					RTCL.append(RTCL[-1])
					nonRTCL.append(nonRTCL[-1] + rand_at_nonRT)
					n_RT.append(n_RT[-1])
					n_nonRT.append(n_nonRT[-1] + (1 if SCL[-1] > MC[-1] else 0))
					SCL.append(MC[-1] + rand_st_nonRT) 
					status.append(2)
					preempt.append(preempt[-1])
					event = [ RTCL[-1], nonRTCL[-1], SCL[-1] ]

					# Handle arrival nonRT and response nonRT
					at_nonRT.append(MC[-1])
					# If RT execute directly, response time = 0
					if n_nonRT[-1] == 0:
						at_nonRT.pop(0)
						R_nonRT.append(0) 

			# if server idle
			else:
				# Check n_RT first and then check nonRT
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
					
					# Handle arrival RT and response RT
					arrival = at_RT.pop(0)
					R_RT.append(MC[-1] - arrival) 
				else: 
					# Check preempt first, if there is no preempt, do non_RT
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
					#  Exexute next non_RT in queue
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
						
						# Handle arrival RT and response RT
						arrival = at_nonRT.pop(0)
						R_nonRT.append(MC[-1] - arrival)
		# Skip the first batch
		if i == 0:
			continue
		# Add batch mean and 95th percentile into list
		R_RT_mean.append(count_mean(R_RT))
		R_nonRT_mean.append(count_mean(R_nonRT))
		RT_95percent.append(count_percentile(R_RT, 0.95))
		nonRT_95percent.append(count_percentile(R_nonRT, 0.95))
	return R_RT_mean, R_nonRT_mean, RT_95percent, nonRT_95percent		

if __name__ == "__main__":
	# User input params
	mean_at_RT = input("Please enter mean inter-arrival time of RT messages:")
	mean_at_nonRT = input("Please enter mean inter-arrival time of nonRT messages:")
	mean_st_RT = input("Please enter mean service time of RT messages:")
	mean_st_nonRT = input("Please enter mean service time of nonRT messages:")
	batch_number = input("Please enter number of batches:")
	batch_size = input("Please enter size of batch:")

	# Get batch results of mean and 95th percentile of nonRT and RT
	R_RT_mean, R_nonRT_mean, RT_95percent, nonRT_95percent = simulation(int(mean_at_RT),int(mean_at_nonRT),int(mean_st_RT),int(mean_st_nonRT),int(batch_number), int(batch_size))
	
	# Count super mean
	RT_batch_mean = count_mean(R_RT_mean)
	nonRT_batch_mean = count_mean(R_nonRT_mean)

	# Count mean standard deviation
	RT_mean_std = count_std(R_RT_mean, RT_batch_mean, batch_number-1)
	nonRT_mean_std = count_std(R_nonRT_mean, nonRT_batch_mean, batch_number-1)
	
	# Count mean confidence intervals
	RT_mean_CI = count_ci(RT_batch_mean, RT_mean_std, batch_number-1)
	nonRT_mean_CI = count_ci(nonRT_batch_mean, nonRT_mean_std, batch_number-1)

	# Count super 95th percentile
	RT_batch_percent = count_percentile(RT_95percent, 0.95)
	nonRT_batch_percent = count_percentile(nonRT_95percent, 0.95)

	# Count 95th percentile standard deviation
	RT_percent_std = count_std(RT_95percent, RT_batch_percent, batch_number-1)
	nonRT_percent_std = count_std(nonRT_95percent, nonRT_batch_percent, batch_number-1)

	# Count mean confidence intervals
	RT_95percent_CI = count_ci(RT_batch_percent, RT_percent_std, batch_number-1)
	nonRT_95percent_CI = count_ci(nonRT_batch_percent, nonRT_percent_std, batch_number-1)

	with open('mean_outputs.csv', mode='a') as csv_file:
		mean_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		mean_writer.writerow([RT_batch_mean, RT_mean_CI[0], RT_mean_CI[1], nonRT_batch_mean, nonRT_mean_CI[0], nonRT_mean_CI[1]])
	
	with open('95thPercentile_outputs.csv', mode='a') as csv_file1:
		percentile_writer = csv.writer(csv_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		percentile_writer.writerow([RT_batch_percent, RT_95percent_CI[0], RT_95percent_CI[1], nonRT_batch_percent, nonRT_95percent_CI[0], nonRT_95percent_CI[1]])

	# Print the results
	print("-----------------------------Results----------------------------------")
	print("RT mean: {}".format(RT_batch_mean))
	print("nonRT mean: {}".format(nonRT_batch_mean))
	print("RT mean CI: {}".format(RT_mean_CI))
	print("nonRT mean CI: {}".format(nonRT_mean_CI))
	print("RT 95th percentile: {}".format(RT_batch_percent))
	print("nonRT 95th percentile: {}".format(nonRT_batch_percent))
	print("RT 95th percentile CI: {}".format(RT_95percent_CI))
	print("nonRT 95th percentile CI: {}".format(nonRT_95percent_CI))




