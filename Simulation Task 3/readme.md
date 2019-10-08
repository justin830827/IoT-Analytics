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
unzip whu24_task3.zip
python ./task3/simulation_task3.py
```
Or
```
unzip task3
cd task3
python ./task3/simulation_task3.py
```

I used the packages which are not included in the eos original environment, if TA would like to run `graph.py` please select `CSC591_ADBI_v3` in the VCL reservation portal, which is another course I took but already had most of package installed, and run the instructions as below:
```
python3 graph.py
```
After that, `mean_plot.png` and `95thpercentile.png` in the folder. As for the data for plotting the graph, I used the data with various MIAT nonRT from requirement and store into `*.csv`.

p.s. The zip file included the original data for plotting, note that the csv will be added more data everytime when the `simulation_task3.py` run. 

p.p.s. Actually my code should be compatible with python > 2.7 and python3 > 3.5, so it should not have problem with running code on `CSC591_ADBI_v3` VCL. However, I think that python3 has better compatibility with `matplotlib`, which is the package I used for plotting, so that I choose run `graph.py` with `python3` instead of `python`. I tried many ways to setup `matplotlib` in python 2.7 environment but it requires admin permission. I would suggest that instructors can setup an environment in VCL and allow student to reserve and test it to ensure both testing results are consistent. 