This readme file is for the instructions to run the script properly.


Connect to the eos.
```
ssh <Unity_ID>@remote-linux.eos.ncsu.edu
```
Copy file to eos, scp source targer or git clone
```
scp whu24_Task3.zip <Unity_id>@remote-linux.eos.ncsu.edu:/afs/u​nity.ncsu.edu/users/<first letter of unity_id>/<unity_id>/
```
To run the script unzip the zipfile and use python compiler
```
unzip whu24_Task3.zip
python ./task3/simulation_task3.py
```
Or
```
unzip task3
cd task3
python simulation_task3.py
```
I used `matplotlib` package for ploting the results as required, but it is not originally installed on the eos server. To make the `graph.py` compiled correctly, it requires to run on VCL Ubuntu 18.04 and intall the `pip` and run `pip install -r requirement.txt`.
Please note that the default seed value is 100, update the seed value manaually to match the test condition. Thanks!
The results will display on terminal as well as store into `mean_outputs.csv` and `95thPercentile_outputs.csv` for ploting the graph. 