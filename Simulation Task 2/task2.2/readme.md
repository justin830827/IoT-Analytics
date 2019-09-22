This readme file is for the instructions to run the script properly.
I follow the instructions below to run my code:
Connect to the eos.
```
ssh <Unity_ID>@remote-linux.eos.ncsu.edu
```
Copy file to eos, scp source targer or git clone
```
scp task2.2.zip <Unity_id>@remote-linux.eos.ncsu.edu:/afs/u​nity.ncsu.edu/users/<first letter of unity_id>/<unity_id>/
```
To run the script unzip the zipfile and use python compiler
```
unzip task2.2
python ./task2.2/simulation_task2_2.py
```
Or
```
unzip task2.2
cd task2.2
python simulation_task2_2.py
```
Please note that the default seed value is 100, update the seed value manaually to match the test condition. Thanks!
The result of file will store into `task2.2_output.txt`.