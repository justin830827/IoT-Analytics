This readme file is for the instructions to run the script properly.


Connect to the eos.
```
ssh <Unity_ID>@remote-linux.eos.ncsu.edu
```
Copy file to eos, scp source targer or git clone
```
scp task2.1.zip <Unity_id>@remote-linux.eos.ncsu.edu:/afs/u​nity.ncsu.edu/users/<first letter of unity_id>/<unity_id>/
```
To run the script unzip the zipfile and use python compiler
```
unzip task2.1
python ./task2.1/simulation_task2_1.py
```

Or
```
unzip task2.1
cd task2.1
python simulation_task2_1.py
```

The result of file will store into `task2.1_output.txt`.
The extra two files `task2.1_out_task1.*.txt` are the results from the task 1 hand simulation.
p.s. The current `*output.txt` may be different from the submited *.txt since I used additional package to print pretty table but not sure if it is allowed for this project. However, I still keep the pretty output just for readibility purpose, the results in both tables are equal. 