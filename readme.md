# Classifying Purchase Orders

Classifying purchase order lines

## QuickStart
Select `spark` as the execution environment, and `cmcpospark.py` as the script, and click **Run** button. 

## Exploring results
After running, you can check out the results in **Run History**.  Exploring the **Run History** will allow you to see the correlation between the parameters you entered and the accuracy of the models.  You can get individual run details by clicking a run in the **Run History** report or clicking the name of the run on the Jobs Panel to the right.  In this sample you will have richer results if you have `matplotlib` installed.

## Quick CLI references
If you want to try exercising the Iris sample from the command line, here are some things to try:

First, launch the Command Prompt or Powershell from the **File** menu. Then enter the following commands:

```
# first let's install matplotlib locally
$ pip install matplotlib

# log in to Azure if you haven't done so
$ az login

# kick off many local runs sequentially
Run `cmcpospark.py` PySpark script in a local Docker container.
```
$ az ml experiment submit -c docker-spark cmcpospark.py
```
Create `myhdi` run configuration to point to an HDI cluster
```
$ az ml computetarget attach cluster --name sparkcluster --address srramhdispark-ssh.azurehdinsight.net --username sshuser --password xxxxx

# prepare the environment
$ az ml experiment prepare -c sparkcluster
```

Run in a remote HDInsight cluster:
```
$ az ml experiment submit -c sparkcluster cmcpospark.py
```