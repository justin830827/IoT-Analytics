This readme file is for the instructions to run the script properly.


Connect to the eos.
```
ssh <Unity_ID>@remote-linux.eos.ncsu.edu
```
Copy file to eos, scp source targer or git clone
```
scp whu24_Task2.zip <Unity_id>@remote-linux.eos.ncsu.edu:/afs/u​nity.ncsu.edu/users/<first letter of unity_id>/<unity_id>/
```
To run the script unzip the zipfile and use python compiler
```
unzip whu24_Task2.zip
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
p.s. The current `*output.txt` may be different from the submited *.txt since I used additional package to print pretty table but not sure if it is allowed for this project. However, I still keep the pretty output just for readibility purpose, the results in both tables are equal. 